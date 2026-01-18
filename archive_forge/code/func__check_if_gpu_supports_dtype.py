import gc
import os
from typing import Dict, List, Tuple, Set, Optional
import torch
import torch.distributed
from vllm.config import (CacheConfig, DeviceConfig, ModelConfig,
from vllm.model_executor import set_random_seed
from vllm.model_executor.parallel_utils import cupy_utils
from vllm.model_executor.parallel_utils.communication_op import (
from vllm.model_executor.parallel_utils.custom_all_reduce import init_custom_ar
from vllm.model_executor.parallel_utils.parallel_state import (
from vllm.sequence import SamplerOutput, SequenceGroupMetadata
from vllm.worker.cache_engine import CacheEngine
from vllm.worker.model_runner import ModelRunner
from vllm.lora.request import LoRARequest
from vllm.utils import is_hip
def _check_if_gpu_supports_dtype(torch_dtype: torch.dtype):
    if torch_dtype == torch.bfloat16:
        compute_capability = torch.cuda.get_device_capability()
        if compute_capability[0] < 8:
            gpu_name = torch.cuda.get_device_name()
            raise ValueError(f'Bfloat16 is only supported on GPUs with compute capability of at least 8.0. Your {gpu_name} GPU has compute capability {compute_capability[0]}.{compute_capability[1]}. You can use float16 instead by explicitly setting the`dtype` flag in CLI, for example: --dtype=half.')