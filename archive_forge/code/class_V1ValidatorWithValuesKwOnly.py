from __future__ import annotations as _annotations
from inspect import Parameter, signature
from typing import Any, Dict, Tuple, Union, cast
from pydantic_core import core_schema
from typing_extensions import Protocol
from ..errors import PydanticUserError
from ._decorators import can_be_positional
class V1ValidatorWithValuesKwOnly(Protocol):
    """A validator with keyword only `values` argument, supported for V1 validators and V2 validators."""

    def __call__(self, __value: Any, *, values: dict[str, Any]) -> Any:
        ...