import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
import ray
from ray import cloudpickle
from ray._private import storage
from ray.types import ObjectRef
from ray.workflow.common import (
from ray.workflow.exceptions import WorkflowNotFoundError
from ray.workflow import workflow_context
from ray.workflow import serialization
from ray.workflow import serialization_context
from ray.workflow.workflow_state import WorkflowExecutionState
from ray.workflow.storage import DataLoadError, DataSaveError, KeyNotFoundError
def _locate_output_in_storage(self, task_id: TaskID) -> Optional[TaskID]:
    result = self.inspect_task(task_id)
    while isinstance(result.output_task_id, str):
        task_id = result.output_task_id
        result = self.inspect_task(result.output_task_id)
    if result.output_object_valid:
        return task_id
    return None