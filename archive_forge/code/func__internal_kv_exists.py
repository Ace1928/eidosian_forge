import json
import logging
from concurrent.futures import Future
from typing import TYPE_CHECKING, Any, Callable, List, Optional, Union
from ray._private import ray_option_utils
from ray.util.client.runtime_context import _ClientWorkerPropertyAPI
def _internal_kv_exists(self, key: Union[str, bytes], *, namespace: Optional[Union[str, bytes]]=None) -> bool:
    """Hook for internal_kv._internal_kv_exists."""
    return self.worker.internal_kv_exists(_as_bytes(key), namespace=_as_bytes(namespace))