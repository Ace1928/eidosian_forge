import json
import logging
from concurrent.futures import Future
from typing import TYPE_CHECKING, Any, Callable, List, Optional, Union
from ray._private import ray_option_utils
from ray.util.client.runtime_context import _ClientWorkerPropertyAPI
def _pin_runtime_env_uri(self, uri: str, expiration_s: int) -> None:
    """Hook for internal_kv._pin_runtime_env_uri."""
    return self.worker.pin_runtime_env_uri(uri, expiration_s)