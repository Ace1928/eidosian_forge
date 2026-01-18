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
class JobState(StateSchema, JobDetails if JobDetails is not None else object):
    """The state of the job that's submitted by Ray's Job APIs or driver jobs"""

    def __init__(self, **kwargs):
        JobDetails.__init__(self, **kwargs)

    @classmethod
    def filterable_columns(cls) -> Set[str]:
        return {'job_id', 'type', 'status', 'submission_id'}

    @classmethod
    def humanify(cls, state: dict) -> dict:
        return state

    @classmethod
    def list_columns(cls, detail: bool=False) -> List[str]:
        if not detail:
            return ['job_id', 'submission_id', 'entrypoint', 'type', 'status', 'message', 'error_type', 'driver_info']
        if isinstance(JobDetails, object):
            return []
        return JobDetails.model_fields if hasattr(JobDetails, 'model_fields') else JobDetails.__fields__

    def asdict(self):
        return JobDetails.dict(self)

    @classmethod
    def schema_dict(cls) -> Dict[str, Any]:
        schema_types = cls.schema()['properties']
        return {k: v['type'] for k, v in schema_types.items() if v.get('type') is not None}