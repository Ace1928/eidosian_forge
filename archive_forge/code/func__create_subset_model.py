from __future__ import annotations
import inspect
import uuid
import warnings
from abc import abstractmethod
from functools import partial
from inspect import signature
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple, Type, Union
from langchain_core.callbacks import (
from langchain_core.callbacks.manager import (
from langchain_core.load.serializable import Serializable
from langchain_core.prompts import (
from langchain_core.pydantic_v1 import (
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import (
from langchain_core.runnables.config import run_in_executor
def _create_subset_model(name: str, model: Type[BaseModel], field_names: list) -> Type[BaseModel]:
    """Create a pydantic model with only a subset of model's fields."""
    fields = {}
    for field_name in field_names:
        field = model.__fields__[field_name]
        t = field.outer_type_ if field.required and (not field.allow_none) else Optional[field.outer_type_]
        fields[field_name] = (t, field.field_info)
    rtn = create_model(name, **fields)
    return rtn