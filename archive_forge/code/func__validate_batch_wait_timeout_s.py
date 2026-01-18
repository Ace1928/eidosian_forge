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
def _validate_batch_wait_timeout_s(batch_wait_timeout_s):
    if not isinstance(batch_wait_timeout_s, (float, int)):
        raise TypeError(f'batch_wait_timeout_s must be a float >= 0, got {batch_wait_timeout_s}')
    if batch_wait_timeout_s < 0:
        raise ValueError(f'batch_wait_timeout_s must be a float >= 0, got {batch_wait_timeout_s}')