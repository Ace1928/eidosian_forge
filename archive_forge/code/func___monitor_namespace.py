import asyncio
import logging
import sys
import traceback
from typing import Any, Dict, List, Optional, Tuple, Union
import kubernetes_asyncio  # type: ignore # noqa: F401
import urllib3
from kubernetes_asyncio import watch
from kubernetes_asyncio.client import (  # type: ignore  # noqa: F401
import wandb
from wandb.sdk.launch.agent import LaunchAgent
from wandb.sdk.launch.errors import LaunchError
from wandb.sdk.launch.runner.abstract import State, Status
from wandb.sdk.launch.utils import get_kube_context_and_api_client
def __monitor_namespace(self, namespace: str, custom_resource: Optional[CustomResource]=None) -> None:
    """Start monitoring a namespaces for resources."""
    if (namespace, Resources.PODS) not in self._monitor_tasks:
        self._monitor_tasks[namespace, Resources.PODS] = create_named_task(f'monitor_pods_{namespace}', self._monitor_pods, namespace)
    if custom_resource is not None:
        if (namespace, custom_resource) not in self._monitor_tasks:
            self._monitor_tasks[namespace, custom_resource] = create_named_task(f'monitor_{custom_resource}_{namespace}', self._monitor_crd, namespace, custom_resource=custom_resource)
    elif (namespace, Resources.JOBS) not in self._monitor_tasks:
        self._monitor_tasks[namespace, Resources.JOBS] = create_named_task(f'monitor_jobs_{namespace}', self._monitor_jobs, namespace)