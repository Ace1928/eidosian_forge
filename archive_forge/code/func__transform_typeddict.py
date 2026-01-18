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
def _transform_typeddict(data: Mapping[str, object], expected_type: type) -> Mapping[str, object]:
    result: dict[str, object] = {}
    annotations = get_type_hints(expected_type, include_extras=True)
    for key, value in data.items():
        type_ = annotations.get(key)
        if type_ is None:
            result[key] = value
        else:
            result[_maybe_transform_key(key, type_)] = _transform_recursive(value, annotation=type_)
    return result