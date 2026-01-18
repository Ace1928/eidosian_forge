from __future__ import annotations
from typing import Any
from typing import Callable
from .base import _registrars
from .registry import _ET
from .registry import _EventKey
from .registry import _ListenerFnType
from .. import exc
from .. import util
def _event_key(target: _ET, identifier: str, fn: _ListenerFnType) -> _EventKey[_ET]:
    for evt_cls in _registrars[identifier]:
        tgt = evt_cls._accept_with(target, identifier)
        if tgt is not None:
            return _EventKey(target, identifier, fn, tgt)
    else:
        raise exc.InvalidRequestError("No such event '%s' for target '%s'" % (identifier, target))