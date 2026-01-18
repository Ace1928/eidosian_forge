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
class PlacementGroupState(StateSchema):
    """PlacementGroup State"""
    placement_group_id: str = state_column(filterable=True)
    name: str = state_column(filterable=True)
    creator_job_id: str = state_column(filterable=True)
    state: TypePlacementGroupStatus = state_column(filterable=True)
    bundles: Optional[List[dict]] = state_column(filterable=False, detail=True)
    is_detached: Optional[bool] = state_column(filterable=True, detail=True)
    stats: Optional[dict] = state_column(filterable=False, detail=True)