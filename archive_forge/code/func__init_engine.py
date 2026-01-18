import asyncio
import time
from functools import partial
from typing import (Any, Dict, Iterable, List, Optional, Set, Tuple, Type,
from vllm.lora.request import LoRARequest
from vllm.config import ModelConfig
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.llm_engine import LLMEngine
from vllm.engine.ray_utils import initialize_cluster, ray
from vllm.logger import init_logger
from vllm.outputs import RequestOutput
from vllm.sampling_params import SamplingParams
def _init_engine(self, *args, **kwargs) -> Union[_AsyncLLMEngine, 'ray.ObjectRef']:
    if not self.engine_use_ray:
        engine_class = self._engine_class
    elif self.worker_use_ray:
        engine_class = ray.remote(num_cpus=0)(self._engine_class).remote
    else:
        cache_config = args[1]
        parallel_config = args[2]
        if parallel_config.tensor_parallel_size == 1:
            num_gpus = cache_config.gpu_memory_utilization
        else:
            num_gpus = 1
        engine_class = ray.remote(num_gpus=num_gpus)(self._engine_class).remote
    return engine_class(*args, **kwargs)