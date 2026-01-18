from typing import Dict, List, Iterator, Optional, Tuple, TYPE_CHECKING
import asyncio
import logging
import time
from collections import defaultdict
import ray
from ray.exceptions import RayTaskError, RayError
from ray.workflow.common import (
from ray.workflow.exceptions import WorkflowCancellationError, WorkflowExecutionError
from ray.workflow.task_executor import get_task_executor, _BakedWorkflowInputs
from ray.workflow.workflow_state import (
def _retry_failed_task(self, workflow_id: str, failed_task_id: TaskID, exc: Exception) -> bool:
    state = self._state
    is_application_error = isinstance(exc, RayTaskError)
    options = state.tasks[failed_task_id].options
    if not is_application_error or options.retry_exceptions:
        if state.task_retries[failed_task_id] < options.max_retries:
            state.task_retries[failed_task_id] += 1
            logger.info(f'Retry [{workflow_id}@{failed_task_id}] ({state.task_retries[failed_task_id]}/{options.max_retries})')
            state.construct_scheduling_plan(failed_task_id)
            return True
    return False