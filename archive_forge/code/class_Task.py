import asyncio
from collections import deque, defaultdict
import dataclasses
from dataclasses import field
import logging
from typing import List, Dict, Optional, Set, Deque, Callable
import ray
from ray.workflow.common import (
from ray.workflow.workflow_context import WorkflowTaskContext
@dataclasses.dataclass
class Task:
    """Data class for a workflow task."""
    task_id: str
    options: WorkflowTaskRuntimeOptions
    user_metadata: Dict
    func_body: Optional[Callable]

    def to_dict(self) -> Dict:
        return {'task_id': self.task_id, 'task_options': self.options.to_dict(), 'user_metadata': self.user_metadata}