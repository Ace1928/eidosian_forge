from __future__ import annotations
import inspect
from typing import TYPE_CHECKING, Any, Type, Union, Generic, TypeVar, Callable, cast
from datetime import date, datetime
from typing_extensions import (
import pydantic
import pydantic.generics
from pydantic.fields import FieldInfo
from ._types import (
from ._utils import (
from ._compat import (
from ._constants import RAW_RESPONSE_HEADER
def _strip_raw_response_header(self) -> None:
    if not is_given(self.headers):
        return
    if self.headers.get(RAW_RESPONSE_HEADER):
        self.headers = {**self.headers}
        self.headers.pop(RAW_RESPONSE_HEADER)