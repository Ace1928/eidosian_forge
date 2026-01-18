from __future__ import annotations
from inspect import iscoroutine
from io import BytesIO
from sys import exc_info
from traceback import extract_tb
from types import GeneratorType
from typing import (
from twisted.internet.defer import Deferred, ensureDeferred
from twisted.python.compat import nativeString
from twisted.python.failure import Failure
from twisted.web._stan import CDATA, CharRef, Comment, Tag, slot, voidElements
from twisted.web.error import FlattenerError, UnfilledSlot, UnsupportedType
from twisted.web.iweb import IRenderable, IRequest
def _fork(d: Deferred[T]) -> Deferred[T]:
    """
    Create a new L{Deferred} based on C{d} that will fire and fail with C{d}'s
    result or error, but will not modify C{d}'s callback type.
    """
    d2: Deferred[T] = Deferred(lambda _: d.cancel())

    def callback(result: T) -> T:
        d2.callback(result)
        return result

    def errback(failure: Failure) -> Failure:
        d2.errback(failure)
        return failure
    d.addCallbacks(callback, errback)
    return d2