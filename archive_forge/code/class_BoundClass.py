from __future__ import annotations
from heapq import heappop, heappush
from itertools import count
from types import MethodType
from typing import (
from simpy.events import (
class BoundClass(Generic[T]):
    """Allows classes to behave like methods.

    The ``__get__()`` descriptor is basically identical to
    ``function.__get__()`` and binds the first argument of the ``cls`` to the
    descriptor instance.

    """

    def __init__(self, cls: Type[T]):
        self.cls = cls

    def __get__(self, instance: Optional[BoundClass], owner: Optional[Type[BoundClass]]=None) -> Union[Type[T], MethodType]:
        if instance is None:
            return self.cls
        return MethodType(self.cls, instance)

    @staticmethod
    def bind_early(instance: object) -> None:
        """Bind all :class:`BoundClass` attributes of the *instance's* class
        to the instance itself to increase performance."""
        for name, obj in instance.__class__.__dict__.items():
            if type(obj) is BoundClass:
                bound_class = getattr(instance, name)
                setattr(instance, name, bound_class)