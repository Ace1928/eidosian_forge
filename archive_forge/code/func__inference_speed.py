import random
import timeit
from functools import wraps
from typing import Callable, Optional
from ..configuration_utils import PretrainedConfig
from ..models.auto.modeling_tf_auto import TF_MODEL_MAPPING, TF_MODEL_WITH_LM_HEAD_MAPPING
from ..utils import is_py3nvml_available, is_tf_available, logging
from .benchmark_utils import (
def _inference_speed(self, model_name: str, batch_size: int, sequence_length: int) -> float:
    strategy = self.args.strategy
    if strategy is None:
        raise ValueError('A device strategy has to be initialized before using TensorFlow.')
    _inference = self._prepare_inference_func(model_name, batch_size, sequence_length)
    return self._measure_speed(_inference)