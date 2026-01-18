from __future__ import annotations
import inspect
from dataclasses import dataclass, field
from enum import Enum
from typing import (
from uuid import UUID, uuid4
from langchain_core.pydantic_v1 import BaseModel
class LabelsDict(TypedDict):
    nodes: dict[str, str]
    edges: dict[str, str]