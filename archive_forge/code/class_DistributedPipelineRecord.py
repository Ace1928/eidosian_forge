from threading import Condition
from types import TracebackType
from typing import Dict, List, Optional, Tuple, Type, Union, cast
import torch
from torch import Tensor
from torch.autograd.profiler import record_function
from torch.distributed import rpc
from fairscale.nn.pipe import microbatch
from fairscale.nn.pipe.checkpoint import Checkpointing, TensorOrTensors
from fairscale.nn.pipe.dependency import fork, join
from fairscale.nn.pipe.microbatch import Batch
from fairscale.nn.pipe.stream import as_cuda, current_stream, is_cuda, use_device, use_stream
from fairscale.nn.pipe.worker import Task, create_workers
from .data import DataConsumer
class DistributedPipelineRecord:
    """A class for storing a single mini-batch (consisting of multiple micro-batches) as input to
    a single partition.
    Args:
        device: the local device that runs the partition.
        rank: the rank of the partition in the pipeline.
        chunks: number of micro-batches in a mini-batch
        num_inputs: number of inputs to the partition.
        consumers: list of consumers of outputs of the partition. Each consumer in the list is a tuple
            (remote_partition_rref, input_idx, output_idx) where remote_partition_rref points to a
            remote DistributedPipelineRecord for consumer partiton for this mini-batch. The output number
            output_idx of this partition will be used as the input number input_idx of that partition.
    """
    DataConsumer = Union[DataConsumer[rpc.RRef]]

    def __init__(self, device: torch.device, rank: int, chunks: int, num_inputs: int, num_outputs: Optional[int], consumers: List[DataConsumer]) -> None:
        self.ready_cv = Condition()
        self.tensors: List[List[Optional[Tensor]]] = [[None] * num_inputs for _ in range(chunks)]
        self.recv_events = [[None] * num_inputs for _ in range(chunks)]
        self.batches: List[Optional[Batch]] = [None] * chunks
        if num_outputs is None:
            num_outputs = 1
        self.forwarded_phony: List[List[List[rpc.RRef]]] = [[[] for j in range(num_outputs)] for i in range(chunks)]
        self.consumers = consumers
        self.rank = rank
        self.device = device

    def __getstate__(self) -> Dict:
        return {}

    def feed(self, chunk: int, input_idx: int, input: Tensor) -> Tensor:
        """This function is called remotely to provide individual tensors of a given chunk."""
        if input.device.type == 'cpu':
            input = input.to(self.device)
        cuda_stream = torch.cuda.current_stream(input.device) if input.device.type == 'cuda' else None
        with self.ready_cv:
            assert self.tensors[chunk][input_idx] is None
            input, phony = fork(input)
            self.recv_events[chunk][input_idx] = cuda_stream.record_event() if cuda_stream is not None else None
            self.tensors[chunk][input_idx] = input
            self.ready_cv.notify_all()
        return phony

    def wait_for(self, chunk: int) -> None:
        """Waits until all elements of given chunk is populated in self.tensors.
        Then it constructs self.batches[chunk] if it is not constructed yet.
        """
        with self.ready_cv:
            while self.batches[chunk] is None and any((b is None for b in self.tensors[chunk])):
                self.ready_cv.wait()
            if self.batches[chunk] is None:
                tensors = cast(List[Tensor], self.tensors[chunk])
                self.batches[chunk] = Batch(tuple(tensors), chunk)

    def fence(self, chunk: int) -> None:
        """Prepares micro-batches for computation."""
        if chunk != 0 and self.consumers and (self.rank > 0):
            batch = self.batches[chunk]
            assert batch is not None
            dependant_tensors = list(batch.tensors)
            for remote_ph_list in self.forwarded_phony[chunk - 1]:
                for remote_ph in remote_ph_list:
                    phony = remote_ph.to_here()
                    dependant_tensors[0] = join(dependant_tensors[0], phony)
            self.batches[chunk] = Batch(tuple(dependant_tensors), chunk)

    def sync_stream(self, chunk: int, stream: torch.cuda.Stream) -> None:
        """syncs the stream with cuda events associated with transmission of the chunck to the cuda device."""
        for e in self.recv_events[chunk]:
            if e is not None:
                stream.wait_event(e)

    def forward_results(self, chunk: int) -> None:
        """Forward outputs of processing the chunk in this parition for processing by next partition."""
        for consumer in self.consumers:
            v = self.get_batch(chunk).value[consumer.output_idx]
            self.forwarded_phony[chunk][consumer.output_idx].append(consumer.consumer.remote().feed(chunk, consumer.consumer_input_idx, v))

    def get_batch(self, chunk: int) -> Batch:
        batch = self.batches[chunk]
        assert batch is not None
        return batch