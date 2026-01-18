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
@dataclass
class TaskInspectResult:
    output_object_valid: bool = False
    output_task_id: Optional[TaskID] = None
    args_valid: bool = False
    func_body_valid: bool = False
    workflow_refs: Optional[List[str]] = None
    task_options: Optional[WorkflowTaskRuntimeOptions] = None
    task_raised_exception: bool = False

    def is_recoverable(self) -> bool:
        return self.output_object_valid or self.output_task_id or (self.args_valid and self.workflow_refs is not None and self.func_body_valid)