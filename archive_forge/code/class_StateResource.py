import datetime
import json
import logging
import sys
from abc import ABC
from dataclasses import asdict, field, fields
from enum import Enum, unique
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import ray.dashboard.utils as dashboard_utils
from ray._private.ray_constants import env_integer
from ray.core.generated.common_pb2 import TaskStatus, TaskType
from ray.core.generated.gcs_pb2 import TaskEvents
from ray.util.state.custom_types import (
from ray.util.state.exception import RayStateApiException
from ray.dashboard.modules.job.pydantic_models import JobDetails
from ray._private.pydantic_compat import IS_PYDANTIC_2
@unique
class StateResource(Enum):
    ACTORS = 'actors'
    JOBS = 'jobs'
    PLACEMENT_GROUPS = 'placement_groups'
    NODES = 'nodes'
    WORKERS = 'workers'
    TASKS = 'tasks'
    OBJECTS = 'objects'
    RUNTIME_ENVS = 'runtime_envs'
    CLUSTER_EVENTS = 'cluster_events'