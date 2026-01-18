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
def _compiled_ray_dag(self):
    import pkg_resources
    required_version = '2.9'
    current_version = pkg_resources.get_distribution('ray').version
    if current_version < required_version:
        raise ValueError(f'Ray version {required_version} or greater is required, but found {current_version}')
    from ray.dag import MultiOutputNode, InputNode
    assert self.parallel_config.worker_use_ray
    with InputNode() as input_data:
        forward_dag = MultiOutputNode([worker.execute_model_compiled_dag_remote.bind(input_data) for worker in self.workers])
    return forward_dag.experimental_compile()