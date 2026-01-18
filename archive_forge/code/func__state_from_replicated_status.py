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
def _state_from_replicated_status(status_dict: Dict[str, int]) -> Optional[State]:
    """Infer overall job status from replicated job status for jobsets.

    More info on jobset:
    https://github.com/kubernetes-sigs/jobset/blob/main/docs/concepts/README.md

    This is useful for detecting when jobsets are starting.
    """
    pods_ready = status_dict.get('ready', 0)
    pods_active = status_dict.get('active', 0)
    if pods_ready >= 1:
        return 'running'
    elif pods_active >= 1:
        return 'starting'
    return None