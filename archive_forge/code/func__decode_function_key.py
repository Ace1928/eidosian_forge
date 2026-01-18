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
def _decode_function_key(key: bytes) -> str:
    return RUNTIME_ENV_FUNC_IDENTIFIER + base64.b64encode(key).decode()