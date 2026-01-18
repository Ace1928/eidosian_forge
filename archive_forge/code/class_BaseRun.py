from __future__ import annotations
import datetime
import warnings
from typing import Any, Dict, List, Optional, Type
from uuid import UUID
from langsmith.schemas import RunBase as BaseRunV2
from langsmith.schemas import RunTypeEnum as RunTypeEnumDep
from langchain_core._api import deprecated
from langchain_core.outputs import LLMResult
from langchain_core.pydantic_v1 import BaseModel, Field, root_validator
@deprecated('0.1.0', alternative='Run', removal='0.2.0')
class BaseRun(BaseModel):
    """Base class for Run."""
    uuid: str
    parent_uuid: Optional[str] = None
    start_time: datetime.datetime = Field(default_factory=datetime.datetime.utcnow)
    end_time: datetime.datetime = Field(default_factory=datetime.datetime.utcnow)
    extra: Optional[Dict[str, Any]] = None
    execution_order: int
    child_execution_order: int
    serialized: Dict[str, Any]
    session_id: int
    error: Optional[str] = None