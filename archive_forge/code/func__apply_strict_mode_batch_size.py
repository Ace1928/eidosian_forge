import collections
import os
import time
from dataclasses import dataclass
from typing import (
import numpy as np
import ray
from ray import DynamicObjectRefGenerator
from ray.data._internal.util import _check_pyarrow_version, _truncated_repr
from ray.types import ObjectRef
from ray.util.annotations import DeveloperAPI
import psutil
def _apply_strict_mode_batch_size(given_batch_size: Optional[Union[int, Literal['default']]], use_gpu: bool) -> Optional[int]:
    if use_gpu and (not given_batch_size or given_batch_size == 'default'):
        raise ValueError("`batch_size` must be provided to `map_batches` when requesting GPUs. The optimal batch size depends on the model, data, and GPU used. It is recommended to use the largest batch size that doesn't result in your GPU device running out of memory. You can view the GPU memory usage via the Ray dashboard.")
    elif given_batch_size == 'default':
        return ray.data.context.STRICT_MODE_DEFAULT_BATCH_SIZE
    else:
        return given_batch_size