from __future__ import annotations
import threading
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from enum import Enum
from typing import (
from uuid import UUID
from typing_extensions import TypedDict
from typing_extensions import Literal
class RunWithAnnotationQueueInfo(RunBase):
    """Run schema with annotation queue info."""
    last_reviewed_time: Optional[datetime] = None
    'The last time this run was reviewed.'
    added_at: Optional[datetime] = None
    'The time this run was added to the queue.'