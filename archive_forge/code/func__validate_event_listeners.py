from __future__ import annotations
import datetime
from collections import abc, namedtuple
from typing import TYPE_CHECKING, Any, Mapping, Optional, Sequence
from bson.objectid import ObjectId
from pymongo.hello import Hello, HelloCompat
from pymongo.helpers import _handle_exception
from pymongo.typings import _Address, _DocumentOut
def _validate_event_listeners(option: str, listeners: Sequence[_EventListeners]) -> Sequence[_EventListeners]:
    """Validate event listeners"""
    if not isinstance(listeners, abc.Sequence):
        raise TypeError(f'{option} must be a list or tuple')
    for listener in listeners:
        if not isinstance(listener, _EventListener):
            raise TypeError(f'Listeners for {option} must be either a CommandListener, ServerHeartbeatListener, ServerListener, TopologyListener, or ConnectionPoolListener.')
    return listeners