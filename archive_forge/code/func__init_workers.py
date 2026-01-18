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
def _init_workers(self):
    Worker = self._dispatch_worker()
    assert self.parallel_config.world_size == 1, 'Ray is required if parallel_config.world_size > 1.'
    self.workers: List[Worker] = []
    distributed_init_method = get_distributed_init_method(get_ip(), get_open_port())
    self.driver_worker = Worker(self.model_config, self.parallel_config, self.scheduler_config, self.device_config, local_rank=0, rank=0, distributed_init_method=distributed_init_method, lora_config=self.lora_config, kv_cache_dtype=self.cache_config.cache_dtype, is_driver_worker=True)
    self._run_workers('init_model')
    self._run_workers('load_model')