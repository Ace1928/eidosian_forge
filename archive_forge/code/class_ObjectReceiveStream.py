from __future__ import annotations
from abc import abstractmethod
from collections.abc import Callable
from typing import Any, Generic, TypeVar, Union
from .._core._exceptions import EndOfStream
from .._core._typedattr import TypedAttributeProvider
from ._resources import AsyncResource
from ._tasks import TaskGroup
class ObjectReceiveStream(UnreliableObjectReceiveStream[T_co]):
    """
    A receive message stream which guarantees that messages are received in the same
    order in which they were sent, and that no messages are missed.
    """