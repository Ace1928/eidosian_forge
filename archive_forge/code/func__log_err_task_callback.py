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
def _log_err_task_callback(task: asyncio.Task) -> None:
    """Callback to log exceptions from tasks."""
    exec = task.exception()
    if exec is not None:
        if isinstance(exec, asyncio.CancelledError):
            wandb.termlog(f'Task {task.get_name()} was cancelled')
            return
        name = str(task) if sys.version_info < (3, 8) else task.get_name()
        wandb.termerror(f'Exception in task {name}')
        tb = exec.__traceback__
        tb_str = ''.join(traceback.format_tb(tb))
        wandb.termerror(tb_str)