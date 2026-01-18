from __future__ import annotations
import uuid
import datetime
import contextlib
from enum import Enum
from typing import Dict, Any, List, Optional, Union, Type
from pydantic import Field, BaseModel
from sqlalchemy.types import TypeDecorator, Text, String, VARCHAR
from sqlalchemy.sql import operators
from sqlalchemy.dialects.postgresql import JSONB
from lazyops.utils import create_unique_id, create_timestamp, timer, Json
class PaginationParams(BaseModel):
    """Pagination params for endpoints."""
    offset: int = 0
    limit: Optional[int] = 100
    order_by: Optional[Any] = None