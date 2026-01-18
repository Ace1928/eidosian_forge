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
def _inspect_task(self, task_id: TaskID) -> TaskInspectResult:
    items = self._scan(self._key_task_prefix(task_id), ignore_errors=True)
    keys = set(items)
    if STEP_OUTPUT in keys:
        return TaskInspectResult(output_object_valid=True)
    if STEP_OUTPUTS_METADATA in keys:
        output_task_id = self._locate_output_task_id(task_id)
        return TaskInspectResult(output_task_id=output_task_id)
    try:
        metadata = self._get(self._key_task_input_metadata(task_id), True)
        return TaskInspectResult(args_valid=STEP_ARGS in keys, func_body_valid=STEP_FUNC_BODY in keys, workflow_refs=metadata['workflow_refs'], task_options=WorkflowTaskRuntimeOptions.from_dict(metadata['task_options']), task_raised_exception=STEP_EXCEPTION in keys)
    except Exception:
        return TaskInspectResult(args_valid=STEP_ARGS in keys, func_body_valid=STEP_FUNC_BODY in keys, task_raised_exception=STEP_EXCEPTION in keys)