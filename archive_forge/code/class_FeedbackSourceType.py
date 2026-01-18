from __future__ import annotations
import threading
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from enum import Enum
from typing import (
from uuid import UUID
from typing_extensions import TypedDict
from typing_extensions import Literal
class FeedbackSourceType(Enum):
    """Feedback source type."""
    API = 'api'
    'General feedback submitted from the API.'
    MODEL = 'model'
    'Model-assisted feedback.'