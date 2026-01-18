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
def _broadcast_exception(self, err: Exception):
    for _, futures in self._task_done_callbacks.items():
        for fut in futures:
            if not fut.done():
                fut.set_exception(err)