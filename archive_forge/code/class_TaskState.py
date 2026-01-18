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
class TaskState(StateSchema):
    """Task State"""
    task_id: str = state_column(filterable=True)
    attempt_number: int = state_column(filterable=True)
    name: str = state_column(filterable=True)
    state: TypeTaskStatus = state_column(filterable=True)
    job_id: str = state_column(filterable=True)
    actor_id: Optional[str] = state_column(filterable=True)
    type: TypeTaskType = state_column(filterable=True)
    func_or_class_name: str = state_column(filterable=True)
    parent_task_id: str = state_column(filterable=True)
    node_id: Optional[str] = state_column(filterable=True)
    worker_id: Optional[str] = state_column(filterable=True)
    worker_pid: Optional[int] = state_column(filterable=True)
    error_type: Optional[str] = state_column(filterable=True)
    language: Optional[str] = state_column(detail=True, filterable=True)
    required_resources: Optional[dict] = state_column(detail=True, filterable=False)
    runtime_env_info: Optional[dict] = state_column(detail=True, filterable=False)
    placement_group_id: Optional[str] = state_column(detail=True, filterable=True)
    events: Optional[List[dict]] = state_column(detail=True, filterable=False, format_fn=Humanify.events)
    profiling_data: Optional[dict] = state_column(detail=True, filterable=False)
    creation_time_ms: Optional[int] = state_column(detail=True, filterable=False, format_fn=Humanify.timestamp)
    start_time_ms: Optional[int] = state_column(detail=True, filterable=False, format_fn=Humanify.timestamp)
    end_time_ms: Optional[int] = state_column(detail=True, filterable=False, format_fn=Humanify.timestamp)
    task_log_info: Optional[dict] = state_column(detail=True, filterable=False)
    error_message: Optional[str] = state_column(detail=True, filterable=False)
    is_debugger_paused: Optional[bool] = state_column(detail=True, filterable=True)