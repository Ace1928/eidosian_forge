import contextlib
import time
from typing import Dict, List, Optional, Tuple, Set, Union
import numpy as np
import torch
import torch.nn as nn
from vllm.config import (DeviceConfig, ModelConfig, LoRAConfig, ParallelConfig,
from vllm.logger import init_logger
from vllm.model_executor import get_model, InputMetadata, SamplingMetadata
from vllm.model_executor.parallel_utils import cupy_utils
from vllm.model_executor.parallel_utils.communication_op import (
from vllm.model_executor.parallel_utils.parallel_state import (
from vllm.model_executor.parallel_utils import custom_all_reduce
from vllm.sampling_params import SamplingParams, SamplingType
from vllm.sequence import SamplerOutput, SequenceData, SequenceGroupMetadata
from vllm.lora.worker_manager import LRUCacheWorkerLoRAManager
from vllm.lora.layers import LoRAMapping
from vllm.lora.request import LoRARequest
from vllm.utils import in_wsl
class CUDAGraphRunner:

    def __init__(self, model: nn.Module):
        self.model = model
        self.graph = None
        self.input_buffers: Dict[str, torch.Tensor] = {}
        self.output_buffers: Dict[str, torch.Tensor] = {}

    def capture(self, input_ids: torch.Tensor, positions: torch.Tensor, kv_caches: List[KVCache], input_metadata: InputMetadata, memory_pool) -> None:
        assert self.graph is None
        with _maybe_cupy_nccl():
            self.model(input_ids, positions, kv_caches, input_metadata)
        torch.cuda.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, pool=memory_pool):
            with _maybe_cupy_nccl():
                hidden_states = self.model(input_ids, positions, kv_caches, input_metadata)
        torch.cuda.synchronize()
        self.input_buffers = {'input_ids': input_ids, 'positions': positions, 'kv_caches': kv_caches, 'slot_mapping': input_metadata.slot_mapping, 'context_lens': input_metadata.context_lens, 'block_tables': input_metadata.block_tables}
        self.output_buffers = {'hidden_states': hidden_states}
        return

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor, kv_caches: List[Tuple[torch.Tensor, torch.Tensor]], input_metadata: InputMetadata) -> torch.Tensor:
        del kv_caches
        self.input_buffers['input_ids'].copy_(input_ids, non_blocking=True)
        self.input_buffers['positions'].copy_(positions, non_blocking=True)
        self.input_buffers['slot_mapping'].copy_(input_metadata.slot_mapping, non_blocking=True)
        self.input_buffers['context_lens'].copy_(input_metadata.context_lens, non_blocking=True)
        self.input_buffers['block_tables'].copy_(input_metadata.block_tables, non_blocking=True)
        self.graph.replay()
        return self.output_buffers['hidden_states']

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)