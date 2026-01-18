import json
import logging
from concurrent.futures import Future
from typing import TYPE_CHECKING, Any, Callable, List, Optional, Union
from ray._private import ray_option_utils
from ray.util.client.runtime_context import _ClientWorkerPropertyAPI
def _converted_key_exists(self, key: str) -> bool:
    """Check if a key UUID is present in the store of converted objects."""
    return self.worker._converted_key_exists(key)