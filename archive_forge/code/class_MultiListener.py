from __future__ import annotations
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Generic, TypeVar
from ..abc import (
@dataclass(eq=False)
class MultiListener(Generic[T_Stream], Listener[T_Stream]):
    """
    Combines multiple listeners into one, serving connections from all of them at once.

    Any MultiListeners in the given collection of listeners will have their listeners
    moved into this one.

    Extra attributes are provided from each listener, with each successive listener
    overriding any conflicting attributes from the previous one.

    :param listeners: listeners to serve
    :type listeners: Sequence[Listener[T_Stream]]
    """
    listeners: Sequence[Listener[T_Stream]]

    def __post_init__(self) -> None:
        listeners: list[Listener[T_Stream]] = []
        for listener in self.listeners:
            if isinstance(listener, MultiListener):
                listeners.extend(listener.listeners)
                del listener.listeners[:]
            else:
                listeners.append(listener)
        self.listeners = listeners

    async def serve(self, handler: Callable[[T_Stream], Any], task_group: TaskGroup | None=None) -> None:
        from .. import create_task_group
        async with create_task_group() as tg:
            for listener in self.listeners:
                tg.start_soon(listener.serve, handler, task_group)

    async def aclose(self) -> None:
        for listener in self.listeners:
            await listener.aclose()

    @property
    def extra_attributes(self) -> Mapping[Any, Callable[[], Any]]:
        attributes: dict = {}
        for listener in self.listeners:
            attributes.update(listener.extra_attributes)
        return attributes