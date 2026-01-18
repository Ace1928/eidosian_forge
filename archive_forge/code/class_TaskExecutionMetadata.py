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
class TaskExecutionMetadata:
    submit_time: Optional[float] = None
    finish_time: Optional[float] = None
    output_size: Optional[int] = None

    @property
    def duration(self):
        return self.finish_time - self.submit_time