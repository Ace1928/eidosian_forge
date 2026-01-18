from __future__ import annotations
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, List, Optional, Sequence, Union
from langchain_core.pydantic_v1 import BaseModel
class FilterDirective(Expr, ABC):
    """A filtering expression."""