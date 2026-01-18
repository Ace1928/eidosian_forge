from typing import Dict, List, Optional, Tuple
import torch
import torch.distributed
from vllm.config import (CacheConfig, DeviceConfig, ModelConfig,
from vllm.model_executor import set_random_seed
from vllm.model_executor.parallel_utils.communication_op import (
from vllm.model_executor.parallel_utils.parallel_state import (
from vllm.sequence import SamplerOutput, SequenceGroupMetadata
from vllm.worker.cache_engine import CacheEngine
from vllm.worker.model_runner import ModelRunner
def _init_distributed_environment(parallel_config: ParallelConfig, rank: int, distributed_init_method: Optional[str]=None, distributed_backend: Optional[str]=None) -> None:
    """Initialize the distributed environment."""
    if torch.distributed.is_initialized():
        torch_world_size = torch.distributed.get_world_size()
        if torch_world_size != parallel_config.world_size:
            raise RuntimeError(f'torch.distributed is already initialized but the torch world size does not match parallel_config.world_size ({torch_world_size} vs. {parallel_config.world_size}).')
    elif not distributed_init_method:
        raise ValueError('distributed_init_method must be set if torch.distributed is not already initialized')
    else:
        distributed_backend = distributed_backend if distributed_backend else 'nccl'
        torch.distributed.init_process_group(backend=distributed_backend, world_size=parallel_config.world_size, rank=rank, init_method=distributed_init_method)
    torch.distributed.all_reduce(torch.zeros(1))
    ensure_model_parallel_initialized(parallel_config.tensor_parallel_size, parallel_config.pipeline_parallel_size)