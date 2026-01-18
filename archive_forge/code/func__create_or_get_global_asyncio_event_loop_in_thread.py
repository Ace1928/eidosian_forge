import asyncio
import concurrent.futures
import threading
import warnings
from dataclasses import dataclass
from typing import Any, AsyncIterator, Dict, Iterator, Optional, Tuple, Union
import ray
from ray import serve
from ray._raylet import GcsClient, ObjectRefGenerator
from ray.serve._private.common import DeploymentID, RequestProtocol
from ray.serve._private.default_impl import create_cluster_node_info_cache
from ray.serve._private.router import RequestMetadata, Router
from ray.serve._private.usage import ServeUsageTag
from ray.serve._private.utils import (
from ray.util import metrics
from ray.util.annotations import Deprecated, DeveloperAPI, PublicAPI
def _create_or_get_global_asyncio_event_loop_in_thread():
    """Provides a global singleton asyncio event loop running in a daemon thread.

    Thread-safe.
    """
    global _global_async_loop
    if _global_async_loop is None:
        with _global_async_loop_creation_lock:
            if _global_async_loop is not None:
                return _global_async_loop
            _global_async_loop = asyncio.new_event_loop()
            thread = threading.Thread(daemon=True, target=_global_async_loop.run_forever)
            thread.start()
    return _global_async_loop