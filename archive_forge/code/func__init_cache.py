import copy
from collections import defaultdict
import os
import time
import pickle
import importlib
from typing import (TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Tuple,
from vllm.lora.request import LoRARequest
from vllm.config import (CacheConfig, DeviceConfig, ModelConfig,
from vllm.core.scheduler import Scheduler, SchedulerOutputs
from vllm.engine.arg_utils import EngineArgs
from vllm.engine.metrics import StatLogger, Stats
from vllm.engine.ray_utils import RayWorkerVllm, initialize_cluster, ray
from vllm.logger import init_logger
from vllm.outputs import RequestOutput
from vllm.sampling_params import SamplingParams
from vllm.sequence import (SamplerOutput, Sequence, SequenceGroup,
from vllm.transformers_utils.tokenizer import (detokenize_incrementally,
from vllm.utils import (Counter, set_cuda_visible_devices, get_ip,
def _init_cache(self) -> None:
    """Profiles the memory usage and initializes the KV cache.

        The engine will first conduct a profiling of the existing memory usage.
        Then, it calculate the maximum possible number of GPU and CPU blocks
        that can be allocated with the remaining free memory.
        More details can be found in the
        :meth:`~vllm.worker.worker.Worker.profile_num_available_blocks` method
        from class :class:`~vllm.worker.Worker`.

        Afterwards, as there may be multiple workers,
        we take the minimum number of blocks across all workers
        to ensure this can be applied to all of them.

        Finally, the engine will initialize the KV cache
        with the calculated number of blocks.

        .. tip::
            You may limit the usage of GPU memory
            by adjusting the `gpu_memory_utilization` parameters.
        """
    num_blocks = self._run_workers('profile_num_available_blocks', block_size=self.cache_config.block_size, gpu_memory_utilization=self.cache_config.gpu_memory_utilization, cpu_swap_space=self.cache_config.swap_space_bytes, cache_dtype=self.cache_config.cache_dtype)
    num_gpu_blocks = min((b[0] for b in num_blocks))
    num_cpu_blocks = min((b[1] for b in num_blocks))
    logger.info(f'# GPU blocks: {num_gpu_blocks}, # CPU blocks: {num_cpu_blocks}')
    if num_gpu_blocks <= 0:
        raise ValueError('No available memory for the cache blocks. Try increasing `gpu_memory_utilization` when initializing the engine.')
    max_seq_len = self.cache_config.block_size * num_gpu_blocks
    if self.model_config.max_model_len > max_seq_len:
        raise ValueError(f"The model's max seq len ({self.model_config.max_model_len}) is larger than the maximum number of tokens that can be stored in KV cache ({max_seq_len}). Try increasing `gpu_memory_utilization` or decreasing `max_model_len` when initializing the engine.")
    self.cache_config.num_gpu_blocks = num_gpu_blocks
    self.cache_config.num_cpu_blocks = num_cpu_blocks
    self._run_workers('init_cache_engine', cache_config=self.cache_config)
    self._run_workers('warm_up_model')