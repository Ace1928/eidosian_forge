import asyncio
import copy
import json
import logging
import os
import psutil
import random
import signal
import string
import subprocess
import sys
import time
import traceback
from asyncio.tasks import FIRST_COMPLETED
from collections import deque
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union
from ray.util.scheduling_strategies import (
import ray
from ray._private.gcs_utils import GcsAioClient
from ray._private.utils import run_background_task
import ray._private.ray_constants as ray_constants
from ray._private.runtime_env.constants import RAY_JOB_CONFIG_JSON_ENV_VAR
from ray.actor import ActorHandle
from ray.dashboard.consts import (
from ray.dashboard.modules.job.common import (
from ray.dashboard.modules.job.utils import file_tail_iterator
from ray.exceptions import ActorUnschedulableError, RuntimeEnvSetupError
from ray.job_submission import JobStatus
from ray._private.event.event_logger import get_event_logger
from ray.core.generated.event_pb2 import Event
def _get_driver_runtime_env(self, resources_specified: bool=False) -> Dict[str, Any]:
    """Get the runtime env that should be set in the job driver.

        Args:
            resources_specified: Whether the user specified resources (CPUs, GPUs,
                custom resources) in the submit_job request. If so, we will skip
                the workaround for GPU detection introduced in #24546, so that the
                behavior matches that of the user specifying resources for any
                other actor.

        Returns:
            The runtime env that should be set in the job driver.
        """
    curr_runtime_env = dict(ray.get_runtime_context().runtime_env)
    if resources_specified:
        return curr_runtime_env
    env_vars = curr_runtime_env.get('env_vars', {})
    env_vars.pop(ray_constants.NOSET_CUDA_VISIBLE_DEVICES_ENV_VAR)
    env_vars.pop(ray_constants.RAY_WORKER_NICENESS)
    curr_runtime_env['env_vars'] = env_vars
    return curr_runtime_env