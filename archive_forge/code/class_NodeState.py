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
class NodeState(StateSchema):
    """Node State"""
    node_id: str = state_column(filterable=True)
    node_ip: str = state_column(filterable=True)
    is_head_node: bool = state_column(filterable=True)
    state: TypeNodeStatus = state_column(filterable=True)
    node_name: str = state_column(filterable=True)
    resources_total: dict = state_column(filterable=False, format_fn=Humanify.node_resources)
    labels: dict = state_column(filterable=False)
    start_time_ms: Optional[int] = state_column(filterable=False, detail=True, format_fn=Humanify.timestamp)
    end_time_ms: Optional[int] = state_column(filterable=False, detail=True, format_fn=Humanify.timestamp)