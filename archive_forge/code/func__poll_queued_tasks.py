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
def _poll_queued_tasks(self) -> List[TaskID]:
    tasks = []
    while True:
        task_id = self._state.pop_frontier_to_run()
        if task_id is None:
            break
        tasks.append(task_id)
    return tasks