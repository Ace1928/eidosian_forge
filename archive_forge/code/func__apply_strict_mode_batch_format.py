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
def _apply_strict_mode_batch_format(given_batch_format: Optional[str]) -> str:
    if given_batch_format == 'default':
        given_batch_format = 'numpy'
    if given_batch_format not in VALID_BATCH_FORMATS_STRICT_MODE:
        raise ValueError(f'The given batch format {given_batch_format} is not allowed in Ray 2.5 (must be one of {VALID_BATCH_FORMATS_STRICT_MODE}).')
    return given_batch_format