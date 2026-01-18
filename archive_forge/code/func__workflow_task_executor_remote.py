import time
from dataclasses import dataclass
import logging
from typing import List, Tuple, Any, Dict, Callable, TYPE_CHECKING
import ray
from ray import ObjectRef
from ray._private import signature
from ray.dag import DAGNode
from ray.workflow import workflow_context
from ray.workflow.workflow_context import get_task_status_info
from ray.workflow import serialization_context
from ray.workflow import workflow_storage
from ray.workflow.common import (
from ray.workflow.workflow_state import WorkflowExecutionState
from ray.workflow.workflow_state_from_dag import workflow_state_from_dag
@ray.remote(num_returns=2)
def _workflow_task_executor_remote(func: Callable, context: 'WorkflowTaskContext', job_id: str, task_id: 'TaskID', baked_inputs: '_BakedWorkflowInputs', runtime_options: 'WorkflowTaskRuntimeOptions') -> Any:
    """The remote version of '_workflow_task_executor'."""
    with workflow_context.workflow_logging_context(job_id):
        return _workflow_task_executor(func, context, task_id, baked_inputs, runtime_options)