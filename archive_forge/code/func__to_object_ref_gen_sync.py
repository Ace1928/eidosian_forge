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
@DeveloperAPI
def _to_object_ref_gen_sync(self, _record_telemetry: bool=True, _allow_running_in_asyncio_loop: bool=False) -> ObjectRefGenerator:
    """Advanced API to convert the generator to a Ray `ObjectRefGenerator`.

        This method is a *blocking* call because it will block until the handle call has
        been assigned to a replica actor. If there are many requests in flight and all
        replicas' queues are full, this may be a slow operation.

        From inside a deployment, `_to_object_ref_gen` should be used instead to avoid
        blocking the asyncio event loop.
        """
    return self._to_object_ref_or_gen_sync(_record_telemetry=_record_telemetry, _allow_running_in_asyncio_loop=_allow_running_in_asyncio_loop)