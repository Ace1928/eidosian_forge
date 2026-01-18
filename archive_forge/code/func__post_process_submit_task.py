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
def _post_process_submit_task(self, task_id: TaskID, store: 'WorkflowStorage') -> None:
    """Update dependencies and reference count etc. after task submission."""
    state = self._state
    if task_id in state.continuation_root:
        if state.tasks[task_id].options.checkpoint:
            store.update_continuation_output_link(state.continuation_root[task_id], task_id)
    else:
        for c in state.upstream_dependencies[task_id]:
            state.reference_set[c].remove(task_id)
            if not state.reference_set[c]:
                del state.reference_set[c]
                state.free_outputs.add(c)