import base64
import json
from ray import cloudpickle
from enum import Enum, unique
import hashlib
from typing import Dict, Optional, Any, Tuple
from dataclasses import dataclass
import ray
from ray import ObjectRef
from ray._private.utils import get_or_create_event_loop
from ray.util.annotations import PublicAPI
@dataclass
class WorkflowTaskRuntimeOptions:
    """Options that will affect a workflow task at runtime."""
    task_type: 'TaskType'
    catch_exceptions: bool
    retry_exceptions: bool
    max_retries: int
    checkpoint: CheckpointModeType
    ray_options: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {'task_type': self.task_type, 'max_retries': self.max_retries, 'catch_exceptions': self.catch_exceptions, 'retry_exceptions': self.retry_exceptions, 'checkpoint': self.checkpoint, 'ray_options': self.ray_options}

    @classmethod
    def from_dict(cls, value: Dict[str, Any]):
        return cls(task_type=TaskType[value['task_type']], max_retries=value['max_retries'], catch_exceptions=value['catch_exceptions'], retry_exceptions=value['retry_exceptions'], checkpoint=value['checkpoint'], ray_options=value['ray_options'])