from __future__ import annotations
import io
import base64
import pathlib
from typing import Any, Mapping, TypeVar, cast
from datetime import date, datetime
from typing_extensions import Literal, get_args, override, get_type_hints
import anyio
import pydantic
from ._utils import (
from .._files import is_base64_file_input
from ._typing import (
from .._compat import model_dump, is_typeddict
def _get_annotated_type(type_: type) -> type | None:
    """If the given type is an `Annotated` type then it is returned, if not `None` is returned.

    This also unwraps the type when applicable, e.g. `Required[Annotated[T, ...]]`
    """
    if is_required_type(type_):
        type_ = get_args(type_)[0]
    if is_annotated_type(type_):
        return type_
    return None