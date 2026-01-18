from __future__ import annotations
from abc import abstractmethod
from collections.abc import Callable
from typing import Any, Generic, TypeVar, Union
from .._core._exceptions import EndOfStream
from .._core._typedattr import TypedAttributeProvider
from ._resources import AsyncResource
from ._tasks import TaskGroup
class UnreliableObjectStream(UnreliableObjectReceiveStream[T_Item], UnreliableObjectSendStream[T_Item]):
    """
    A bidirectional message stream which does not guarantee the order or reliability of
    message delivery.
    """