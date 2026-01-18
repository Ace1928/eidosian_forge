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
@lazyproperty
def active_key(self) -> str:
    if self.worker_id:
        return f'{self.queue.active_key}:{self.worker_id}'
    if self.worker_name:
        return f'{self.queue.active_key}:{self.worker_name}'
    return self.queue.active_key