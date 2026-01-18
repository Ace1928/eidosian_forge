from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union
import torch
from torch import Tensor, nn
from torch.distributed import rpc
from fairscale.internal import torch_version
from fairscale.nn.pipe import microbatch
from .data import DataConsumer
from .graph import Node, PipelineModulesGraph
from .partition_handler import DistributedPipelineRecord, PartitionHandler
class DistributedPipeline(nn.Module):
    """Wraps a :class:`PipelineModulesGraph` model to train on using synchronous pipeline
    parallelism. If the model requires lots of memory and doesn't fit on a single GPU,
    pipeline parallelism is a useful technique to employ for training.

    The implementation is based on the torchgpipe_ paper.

    .. _torchgpipe: https://arxiv.org/abs/2004.09910

    PipelineModulesGraph combines pipeline parallelism with checkpointing to reduce peak
    memory required to train while minimizing device under-utilization.

    You should place all the modules on the appropriate rpc workers and devices and wrap
    them into an :class:`PipelineModulesGraph` module defining the connection between them.

    Args:
        module (:class:`PipelineModulesGraph`):
        model to be parallelized using pipelining. Each module
            in the graph has to have all of its parameters on a single
            device.
        chunks (int):
            number of micro-batches (default: ``1``)
        checkpoint (str):
            when to enable checkpointing, one of ``'always'``,
            ``'except_last'``, or ``'never'`` (default: ``'except_last'``).
            ``'never'`` disables checkpointing completely, ``'except_last'``
            enables checkpointing for all micro-batches except the last one
            and ``'always'`` enables checkpointing for all micro-batches.
    """

    @dataclass
    class Partition:
        nodes: List[Node]
        handler: rpc.RRef

        def __hash__(self) -> int:
            return hash(self.handler)
    DataConsumer = DataConsumer[Partition]

    def __init__(self, graph: PipelineModulesGraph, chunks: int=1, checkpoint: str='except_last') -> None:
        super().__init__()
        check_pytorch_version()
        chunks = int(chunks)
        checkpoint = str(checkpoint)
        if chunks <= 0:
            raise ValueError('number of chunks must be positive integer')
        if checkpoint not in ['always', 'except_last', 'never']:
            raise ValueError("checkpoint is not one of 'always', 'except_last', or 'never'")
        self.chunks = chunks
        checkpoint_stop = {'always': self.chunks, 'except_last': self.chunks - 1, 'never': 0}[checkpoint]
        self.partitions = [self.Partition(nodes, rpc.remote(handler.owner(), PartitionHandler, args=(handler, nodes[0].module.device, len(nodes[0].inputs), nodes[-1].num_outputs, i, self.chunks, checkpoint_stop))) for i, (nodes, handler) in enumerate(graph.partition_graph())]
        self.input_consumers = [next((self.DataConsumer(partition, input_consumer.consumer_input_idx, input_consumer.output_idx) for partition in self.partitions if partition.nodes[0] is input_consumer.consumer)) for input_consumer in graph.model_input_consumers]
        self.graph = graph

    def cuda(self, device: Optional[Device]=None) -> 'DistributedPipeline':
        raise MOVING_DENIED

    def cpu(self) -> 'DistributedPipeline':
        raise MOVING_DENIED

    def to(self, *args: Any, **kwargs: Any) -> 'DistributedPipeline':
        if 'device' in kwargs or 'tensor' in kwargs:
            raise MOVING_DENIED
        if args:
            if isinstance(args[0], (torch.device, int, str)):
                raise MOVING_DENIED
            if torch.is_tensor(args[0]):
                raise MOVING_DENIED
        return super().to(*args, **kwargs)

    def parameter_rrefs(self) -> List[rpc.RRef]:
        remote_params = []
        for p in self.partitions:
            remote_params.extend(p.handler.rpc_sync().local_parameter_rrefs())
        return remote_params

    def forward(self, *inputs: Tensor) -> rpc.RRef:
        for i, input in enumerate(inputs):
            microbatch.check(input)
        batches_list = [microbatch.scatter(input, self.chunks) for input in inputs]
        pipeline_records: Dict[DistributedPipeline.Partition, rpc.RRef] = {}
        for partition in reversed(self.partitions):
            r_handler = partition.handler.remote()
            consumers = []
            for consumer in partition.nodes[-1].output_consumers:
                consumer_partition = next((p for p in self.partitions if p.nodes[0] is consumer.consumer))
                assert consumer_partition in pipeline_records
                consumers.append(DistributedPipelineRecord.DataConsumer(pipeline_records[consumer_partition], consumer.consumer_input_idx, consumer.output_idx))
            pipeline_records[partition] = r_handler.make_pipeline_record(consumers)
            this_result = r_handler.run_pipeline(pipeline_records[partition])
            if partition is self.partitions[-1]:
                result = this_result
        for i, b in enumerate(zip(*batches_list)):
            for input_consumer in self.input_consumers:
                pipeline_record = pipeline_records[input_consumer.consumer]
                if pipeline_record.owner().name == rpc.get_worker_info().name:
                    pipeline_record.local_value().feed(i, input_consumer.consumer_input_idx, b[input_consumer.output_idx].value)
                else:
                    pipeline_record.rpc_async().feed(i, input_consumer.consumer_input_idx, b[input_consumer.output_idx].value)
        return result