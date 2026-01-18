from __future__ import annotations
import enum
import typing
import datetime
import croniter
from aiokeydb.v2.types.base import BaseModel, lazyproperty, Field, validator
from aiokeydb.v2.utils.queue import (
from aiokeydb.v2.configs import settings
from aiokeydb.v2.utils.logs import logger
from aiokeydb.v2.types.static import JobStatus, TaskType, TERMINAL_STATUSES, UNSUCCESSFUL_TERMINAL_STATUSES, INCOMPLETE_STATUSES
class JobProgress(BaseModel):
    """
    Holds the progress of a job
    """
    total: int = 0
    completed: int = 0

    @property
    def progress(self) -> float:
        """
        Returns the progress of a job as a float between 0.0 and 1.0
        """
        return 0.0 if self.total == 0 else self.completed / self.total