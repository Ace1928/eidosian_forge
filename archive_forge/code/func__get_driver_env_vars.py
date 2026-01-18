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
def _get_driver_env_vars(self, resources_specified: bool) -> Dict[str, str]:
    """Returns environment variables that should be set in the driver."""
    if ray_constants.RAY_ADDRESS_ENVIRONMENT_VARIABLE in os.environ:
        os.environ[ray_constants.RAY_ADDRESS_ENVIRONMENT_VARIABLE] = 'auto'
    ray_addr = ray._private.services.canonicalize_bootstrap_address_or_die('auto', ray.worker._global_node._ray_params.temp_dir)
    assert ray_addr is not None
    return {RAY_JOB_CONFIG_JSON_ENV_VAR: json.dumps({'runtime_env': self._get_driver_runtime_env(resources_specified), 'metadata': self._metadata}), ray_constants.RAY_ADDRESS_ENVIRONMENT_VARIABLE: ray_addr, 'PYTHONUNBUFFERED': '1'}