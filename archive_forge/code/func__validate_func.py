from __future__ import annotations
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, List, Optional, Sequence, Union
from langchain_core.pydantic_v1 import BaseModel
def _validate_func(self, func: Union[Operator, Comparator]) -> None:
    if isinstance(func, Operator) and self.allowed_operators is not None:
        if func not in self.allowed_operators:
            raise ValueError(f'Received disallowed operator {func}. Allowed comparators are {self.allowed_operators}')
    if isinstance(func, Comparator) and self.allowed_comparators is not None:
        if func not in self.allowed_comparators:
            raise ValueError(f'Received disallowed comparator {func}. Allowed comparators are {self.allowed_comparators}')