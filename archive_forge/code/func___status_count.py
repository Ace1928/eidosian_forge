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
def __status_count(self) -> Dict[State, int]:
    """Get a dictionary mapping statuses to the # monitored jobs with each status."""
    counts = dict()
    for _, status in self._job_states.items():
        state = status.state
        if state not in counts:
            counts[state] = 1
        else:
            counts[state] += 1
    return counts