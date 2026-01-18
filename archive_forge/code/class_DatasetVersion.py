from __future__ import annotations
import threading
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from enum import Enum
from typing import (
from uuid import UUID
from typing_extensions import TypedDict
from typing_extensions import Literal
class DatasetVersion(BaseModel):
    """Class representing a dataset version."""
    tags: Optional[List[str]] = None
    as_of: datetime