from __future__ import annotations
import logging # isort:skip
import weakref
from collections import defaultdict
from functools import wraps
from typing import TYPE_CHECKING, Any, Callable
from ..core.enums import HoldPolicy, HoldPolicyType
from ..events import (
from ..model import Model
from ..models.callbacks import Callback as JSEventCallback
from ..util.callback_manager import _check_callback
from .events import (
from .locking import UnlockedDocumentProxy
def _on_event(self, event: str | type[Event], *callbacks: EventCallback | JSEventCallback) -> None:
    if not isinstance(event, str) and issubclass(event, Event):
        event = event.event_name
    if event not in _CONCRETE_EVENT_CLASSES:
        raise ValueError(f'Unknown event {event}')
    if not issubclass(_CONCRETE_EVENT_CLASSES[event], DocumentEvent):
        raise ValueError('Document.on_event may only be used to subscribe to events of type DocumentEvent. To subscribe to a ModelEvent use the Model.on_event method.')
    for callback in callbacks:
        if isinstance(callback, JSEventCallback):
            self._js_event_callbacks[event].append(callback)
        else:
            _check_callback(callback, ('event',), what='Event callback')
            self._event_callbacks[event].append(callback)