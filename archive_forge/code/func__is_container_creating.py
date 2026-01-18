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
def _is_container_creating(status: 'V1PodStatus') -> bool:
    """Check if this pod has started creating containers."""
    for container_status in status.container_statuses or []:
        if container_status.state and container_status.state.waiting and (container_status.state.waiting.reason == 'ContainerCreating'):
            return True
    return False