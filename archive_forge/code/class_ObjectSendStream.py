from __future__ import annotations
from abc import abstractmethod
from collections.abc import Callable
from typing import Any, Generic, TypeVar, Union
from .._core._exceptions import EndOfStream
from .._core._typedattr import TypedAttributeProvider
from ._resources import AsyncResource
from ._tasks import TaskGroup
class ObjectSendStream(UnreliableObjectSendStream[T_contra]):
    """
    A send message stream which guarantees that messages are delivered in the same order
    in which they were sent, without missing any messages in the middle.
    """