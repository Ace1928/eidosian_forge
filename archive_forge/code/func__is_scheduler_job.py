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
def _is_scheduler_job(run_spec: Dict[str, Any]) -> bool:
    """Determine whether a job/runSpec is a sweep scheduler."""
    if not run_spec:
        _logger.debug('Recieved runSpec in _is_scheduler_job that was empty')
    if run_spec.get('uri') != Scheduler.PLACEHOLDER_URI:
        return False
    if run_spec.get('resource') == 'local-process':
        if run_spec.get('job'):
            return True
        cmd = run_spec.get('overrides', {}).get('entry_point', [])
        if len(cmd) < 3:
            return False
        if cmd[:2] != ['wandb', 'scheduler']:
            return False
    return True