from __future__ import annotations
import contextlib
import enum
import inspect
import os
import re
import sys
import types
import typing as t
from ast import literal_eval
from .utils.bunch import Bunch
from .utils.descriptions import add_article, class_of, describe, repr_type
from .utils.getargspec import getargspec
from .utils.importstring import import_item
from .utils.sentinel import Sentinel
from .utils.warnings import deprecated_method, should_warn, warn
from all trait attributes.
def _notify_observers(self, event: Bunch) -> None:
    """Notify observers of any event"""
    if not isinstance(event, Bunch):
        event = Bunch(event)
    name, type = (event['name'], event['type'])
    callables = []
    if name in self._trait_notifiers:
        callables.extend(self._trait_notifiers.get(name, {}).get(type, []))
        callables.extend(self._trait_notifiers.get(name, {}).get(All, []))
    if All in self._trait_notifiers:
        callables.extend(self._trait_notifiers.get(All, {}).get(type, []))
        callables.extend(self._trait_notifiers.get(All, {}).get(All, []))
    magic_name = '_%s_changed' % name
    if event['type'] == 'change' and hasattr(self, magic_name):
        class_value = getattr(self.__class__, magic_name)
        if not isinstance(class_value, ObserveHandler):
            deprecated_method(class_value, self.__class__, magic_name, 'use @observe and @unobserve instead.')
            cb = getattr(self, magic_name)
            if cb not in callables:
                callables.append(_callback_wrapper(cb))
    for c in callables:
        if isinstance(c, _CallbackWrapper):
            c = c.__call__
        elif isinstance(c, EventHandler) and c.name is not None:
            c = getattr(self, c.name)
        c(event)