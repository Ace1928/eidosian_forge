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
def _get_or_create_router(self) -> Union[Router, asyncio.AbstractEventLoop]:
    if self._router is None:
        node_id = ray.get_runtime_context().get_node_id()
        try:
            cluster_node_info_cache = create_cluster_node_info_cache(GcsClient(address=ray.get_runtime_context().gcs_address))
            cluster_node_info_cache.update()
            availability_zone = cluster_node_info_cache.get_node_az(node_id)
        except Exception:
            availability_zone = None
        self._router = Router(serve.context._get_global_client()._controller, self.deployment_id, node_id, get_current_actor_id(), availability_zone, event_loop=_create_or_get_global_asyncio_event_loop_in_thread(), _prefer_local_node_routing=self.handle_options._prefer_local_routing, _router_cls=self.handle_options._router_cls)
    return (self._router, self._router._event_loop)