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
class RuntimeEnvState(StateSchema):
    """Runtime Environment State"""
    runtime_env: dict = state_column(filterable=True)
    success: bool = state_column(filterable=True)
    creation_time_ms: Optional[float] = state_column(filterable=False, format_fn=Humanify.timestamp)
    node_id: str = state_column(filterable=True)
    ref_cnt: Optional[int] = state_column(detail=True, filterable=False)
    error: Optional[str] = state_column(detail=True, filterable=True)