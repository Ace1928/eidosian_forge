import atexit
import logging
from functools import partial
from types import FunctionType
from typing import Callable, Optional, Type, Union
import ray
import ray.cloudpickle as pickle
from ray.experimental.internal_kv import (
from ray.tune.error import TuneError
from ray.util.annotations import DeveloperAPI
def _make_key(prefix: str, category: str, key: str):
    """Generate a binary key for the given category and key.

    Args:
        prefix: Prefix
        category: The category of the item
        key: The unique identifier for the item

    Returns:
        The key to use for storing a the value.
    """
    return b'TuneRegistry:' + prefix.encode('ascii') + b':' + category.encode('ascii') + b'/' + key.encode('ascii')