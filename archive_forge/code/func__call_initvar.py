from __future__ import annotations as _annotations
import dataclasses
import sys
import types
from typing import TYPE_CHECKING, Any, Callable, Generic, NoReturn, TypeVar, overload
from typing_extensions import Literal, TypeGuard, dataclass_transform
from ._internal import _config, _decorators, _typing_extra
from ._internal import _dataclasses as _pydantic_dataclasses
from ._migration import getattr_migration
from .config import ConfigDict
from .fields import Field, FieldInfo
def _call_initvar(*args: Any, **kwargs: Any) -> NoReturn:
    """This function does nothing but raise an error that is as similar as possible to what you'd get
        if you were to try calling `InitVar[int]()` without this monkeypatch. The whole purpose is just
        to ensure typing._type_check does not error if the type hint evaluates to `InitVar[<parameter>]`.
        """
    raise TypeError("'InitVar' object is not callable")