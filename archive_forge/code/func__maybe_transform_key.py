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
def _maybe_transform_key(key: str, type_: type) -> str:
    """Transform the given `data` based on the annotations provided in `type_`.

    Note: this function only looks at `Annotated` types that contain `PropertInfo` metadata.
    """
    annotated_type = _get_annotated_type(type_)
    if annotated_type is None:
        return key
    annotations = get_args(annotated_type)[1:]
    for annotation in annotations:
        if isinstance(annotation, PropertyInfo) and annotation.alias is not None:
            return annotation.alias
    return key