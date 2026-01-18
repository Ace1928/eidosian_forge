from __future__ import annotations
import logging # isort:skip
from collections import defaultdict
from inspect import signature
from typing import (
from ..events import Event, ModelEvent
from ..util.functions import get_param_info
class EventCallbackManager:
    """ A mixin class to provide an interface for registering and
    triggering event callbacks on the Python side.

    """
    document: Document | None
    id: ID
    subscribed_events: set[str]
    _event_callbacks: dict[str, list[EventCallback]]

    def __init__(self, *args: Any, **kw: Any) -> None:
        super().__init__(*args, **kw)
        self._event_callbacks = defaultdict(list)

    def on_event(self, event: str | type[Event], *callbacks: EventCallback) -> None:
        """ Run callbacks when the specified event occurs on this Model

        Not all Events are supported for all Models.
        See specific Events in :ref:`bokeh.events` for more information on
        which Models are able to trigger them.
        """
        if not isinstance(event, str) and issubclass(event, Event):
            event = event.event_name
        for callback in callbacks:
            if _nargs(callback) != 0:
                _check_callback(callback, ('event',), what='Event callback')
            self._event_callbacks[event].append(callback)
        self.subscribed_events.add(event)

    def _trigger_event(self, event: ModelEvent) -> None:

        def invoke() -> None:
            for callback in self._event_callbacks.get(event.event_name, []):
                if event.model is not None and self.id == event.model.id:
                    if _nargs(callback) == 0:
                        cast(EventCallbackWithoutEvent, callback)()
                    else:
                        cast(EventCallbackWithEvent, callback)(event)
        if self.document is not None:
            from ..model import Model
            self.document.callbacks.notify_event(cast(Model, self), event, invoke)
        else:
            invoke()

    def _update_event_callbacks(self) -> None:
        if self.document is None:
            return
        for key in self._event_callbacks:
            from ..model import Model
            self.document.callbacks.subscribe(key, cast(Model, self))