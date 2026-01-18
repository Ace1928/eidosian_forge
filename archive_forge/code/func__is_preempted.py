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
def _is_preempted(status: 'V1PodStatus') -> bool:
    """Check if this pod has been preempted."""
    if hasattr(status, 'conditions') and status.conditions is not None:
        for condition in status.conditions:
            if condition.type == 'DisruptionTarget' and condition.reason in ['EvictionByEvictionAPI', 'PreemptionByScheduler', 'TerminationByKubelet']:
                return True
    return False