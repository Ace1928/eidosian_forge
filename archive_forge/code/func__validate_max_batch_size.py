import asyncio
import time
from dataclasses import dataclass
from functools import wraps
from inspect import isasyncgenfunction, iscoroutinefunction
from typing import (
from ray._private.signature import extract_signature, flatten_args, recover_args
from ray._private.utils import get_or_create_event_loop
from ray.serve._private.utils import extract_self_if_method_call
from ray.serve.exceptions import RayServeException
from ray.util.annotations import PublicAPI
def _validate_max_batch_size(max_batch_size):
    if not isinstance(max_batch_size, int):
        if isinstance(max_batch_size, float) and max_batch_size.is_integer():
            max_batch_size = int(max_batch_size)
        else:
            raise TypeError(f'max_batch_size must be integer >= 1, got {max_batch_size}')
    if max_batch_size < 1:
        raise ValueError(f'max_batch_size must be an integer >= 1, got {max_batch_size}')