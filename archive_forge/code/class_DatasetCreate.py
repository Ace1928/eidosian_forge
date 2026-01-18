from __future__ import annotations
import threading
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from enum import Enum
from typing import (
from uuid import UUID
from typing_extensions import TypedDict
from typing_extensions import Literal
class DatasetCreate(DatasetBase):
    """Dataset create model."""
    id: Optional[UUID] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))