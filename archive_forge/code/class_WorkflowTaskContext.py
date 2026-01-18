import logging
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Optional
import ray
from ray._private.ray_logging import configure_log_file, get_worker_log_file_name
from ray.workflow.common import CheckpointModeType, WorkflowStatus
@dataclass
class WorkflowTaskContext:
    """
    The structure for saving workflow task context. The context provides
    critical info (e.g. where to checkpoint, which is its parent task)
    for the task to execute correctly.
    """
    workflow_id: Optional[str] = None
    task_id: str = ''
    creator_task_id: str = ''
    checkpoint: CheckpointModeType = True
    catch_exceptions: bool = False