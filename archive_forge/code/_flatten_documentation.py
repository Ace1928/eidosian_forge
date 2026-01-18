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

    Collate a string representation of C{root} into a single string.

    This is basically gluing L{flatten} to an L{io.BytesIO} and returning
    the results. See L{flatten} for the exact meanings of C{request} and
    C{root}.

    @return: A L{Deferred} which will be called back with a single UTF-8 encoded
        string as its result when C{root} has been completely flattened or which
        will be errbacked if an unexpected exception occurs.
    