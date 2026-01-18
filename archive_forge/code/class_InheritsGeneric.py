from __future__ import annotations
from os import PathLike
from typing import (
from typing_extensions import Literal, Protocol, TypeAlias, TypedDict, override, runtime_checkable
import httpx
import pydantic
from httpx import URL, Proxy, Timeout, Response, BaseTransport, AsyncBaseTransport
@runtime_checkable
class InheritsGeneric(Protocol):
    """Represents a type that has inherited from `Generic`

    The `__orig_bases__` property can be used to determine the resolved
    type variable for a given base class.
    """
    __orig_bases__: tuple[_GenericAlias]