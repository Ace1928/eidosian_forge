import asyncio
import logging
import os
import pprint
import threading
import time
import traceback
from dataclasses import dataclass
from multiprocessing import Event
from typing import Any, Dict, List, Optional, Union
import wandb
from wandb.apis.internal import Api
from wandb.errors import CommError
from wandb.sdk.launch._launch_add import launch_add
from wandb.sdk.launch.runner.local_container import LocalSubmittedRun
from wandb.sdk.launch.runner.local_process import LocalProcessRunner
from wandb.sdk.launch.sweeps.scheduler import Scheduler
from wandb.sdk.lib import runid
from .. import loader
from .._project_spec import LaunchProject
from ..builder.build import construct_agent_configs
from ..errors import LaunchDockerError, LaunchError
from ..utils import (
from .job_status_tracker import JobAndRunStatusTracker
from .run_queue_item_file_saver import RunQueueItemFileSaver
def _assert_secure(self, launch_spec: Dict[str, Any]) -> None:
    """If secure mode is set, make sure no vulnerable keys are overridden."""
    if not self._secure_mode:
        return
    k8s_config = launch_spec.get('resource_args', {}).get('kubernetes', {})
    pod_secure_keys = ['hostPID', 'hostIPC', 'hostNetwork', 'initContainers']
    pod_spec = k8s_config.get('spec', {}).get('template', {}).get('spec', {})
    for key in pod_secure_keys:
        if key in pod_spec:
            raise ValueError(f'This agent is configured to lock "{key}" in pod spec but the job specification attempts to override it.')
    container_specs = pod_spec.get('containers', [])
    for container_spec in container_specs:
        if 'command' in container_spec:
            raise ValueError('This agent is configured to lock "command" in container spec but the job specification attempts to override it.')
    if launch_spec.get('overrides', {}).get('entry_point'):
        raise ValueError('This agent is configured to lock the "entrypoint" override but the job specification attempts to override it.')