from __future__ import annotations
import threading
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from enum import Enum
from typing import (
from uuid import UUID
from typing_extensions import TypedDict
from typing_extensions import Literal
class BatchIngestConfig(TypedDict, total=False):
    """Configuration for batch ingestion.

    Attributes:
        scale_up_qsize_trigger (int): The queue size threshold that triggers scaling up.
        scale_up_nthreads_limit (int): The maximum number of threads to scale up to.
        scale_down_nempty_trigger (int): The number of empty threads that triggers
            scaling down.
        size_limit (int): The maximum size limit for the batch.
    """
    scale_up_qsize_trigger: int
    scale_up_nthreads_limit: int
    scale_down_nempty_trigger: int
    size_limit: int
    size_limit_bytes: Optional[int]