from __future__ import annotations
import threading
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from enum import Enum
from typing import (
from uuid import UUID
from typing_extensions import TypedDict
from typing_extensions import Literal
class FeedbackSourceBase(BaseModel):
    """Base class for feedback sources.

    This represents whether feedback is submitted from the API, model, human labeler,
        etc.

    Attributes:
        type (str): The type of the feedback source.
        metadata (Optional[Dict[str, Any]]): Additional metadata for the feedback
            source.
    """
    type: str
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)