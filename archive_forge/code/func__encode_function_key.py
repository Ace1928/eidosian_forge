import traceback
import logging
import base64
import os
from typing import Dict, Any, Callable, Union, Optional
import ray
import ray._private.ray_constants as ray_constants
from ray._private.storage import _load_class
import ray.cloudpickle as pickle
from ray.runtime_env import RuntimeEnv
def _encode_function_key(key: str) -> bytes:
    assert key.startswith(RUNTIME_ENV_FUNC_IDENTIFIER)
    return base64.b64decode(key[len(RUNTIME_ENV_FUNC_IDENTIFIER):])