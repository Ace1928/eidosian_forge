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
@dataclass(init=not IS_PYDANTIC_2)
class ObjectState(StateSchema):
    """Object State"""
    object_id: str = state_column(filterable=True)
    object_size: int = state_column(filterable=True, format_fn=Humanify.memory)
    task_status: TypeTaskStatus = state_column(filterable=True)
    reference_type: TypeReferenceType = state_column(filterable=True)
    call_site: str = state_column(filterable=True)
    type: TypeWorkerType = state_column(filterable=True)
    pid: int = state_column(filterable=True)
    ip: str = state_column(filterable=True)