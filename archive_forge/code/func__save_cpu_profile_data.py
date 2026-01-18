import asyncio
import inspect
import logging
import os
import pickle
import time
import traceback
from contextlib import asynccontextmanager
from importlib import import_module
from typing import Any, AsyncGenerator, Callable, Dict, Optional, Tuple
import aiorwlock
import starlette.responses
from starlette.requests import Request
from starlette.types import Message, Receive, Scope, Send
import ray
from ray import cloudpickle
from ray._private.async_compat import sync_to_async
from ray._private.utils import get_or_create_event_loop
from ray.actor import ActorClass, ActorHandle
from ray.remote_function import RemoteFunction
from ray.serve import metrics
from ray.serve._private.autoscaling_metrics import InMemoryMetricsStore
from ray.serve._private.common import (
from ray.serve._private.config import DeploymentConfig
from ray.serve._private.constants import (
from ray.serve._private.deployment_info import CONTROL_PLANE_CONCURRENCY_GROUP
from ray.serve._private.http_util import (
from ray.serve._private.logging_utils import (
from ray.serve._private.router import RequestMetadata
from ray.serve._private.utils import (
from ray.serve._private.version import DeploymentVersion
from ray.serve.deployment import Deployment
from ray.serve.exceptions import RayServeException
from ray.serve.grpc_util import RayServegRPCContext
from ray.serve.schema import LoggingConfig
def _save_cpu_profile_data(self) -> str:
    """Saves CPU profiling data, if CPU profiling is enabled.

            Logs a warning if CPU profiling is disabled.
            """
    if self.cpu_profiler is not None:
        import marshal
        self.cpu_profiler.snapshot_stats()
        with open(self.cpu_profiler_log, 'wb') as f:
            marshal.dump(self.cpu_profiler.stats, f)
        logger.info(f'Saved CPU profile data to file "{self.cpu_profiler_log}"')
        return self.cpu_profiler_log
    else:
        logger.error('Attempted to save CPU profile data, but failed because no CPU profiler was running! Enable CPU profiling by enabling the RAY_SERVE_ENABLE_CPU_PROFILING env var.')