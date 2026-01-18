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
Transform the given data against the expected type.

    Args:
        annotation: The direct type annotation given to the particular piece of data.
            This may or may not be wrapped in metadata types, e.g. `Required[T]`, `Annotated[T, ...]` etc

        inner_type: If applicable, this is the "inside" type. This is useful in certain cases where the outside type
            is a container type such as `List[T]`. In that case `inner_type` should be set to `T` so that each entry in
            the list can be transformed using the metadata from the container type.

            Defaults to the same value as the `annotation` argument.
    