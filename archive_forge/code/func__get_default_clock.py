from __future__ import annotations
from typing import Callable, Iterable, Optional, cast
from zope.interface import directlyProvides, implementer, providedBy
from OpenSSL.SSL import Connection, Error, SysCallError, WantReadError, ZeroReturnError
from twisted.internet._producer_helpers import _PullToPush
from twisted.internet._sslverify import _setAcceptableProtocols
from twisted.internet.interfaces import (
from twisted.internet.main import CONNECTION_LOST
from twisted.internet.protocol import Protocol
from twisted.protocols.policies import ProtocolWrapper, WrappingFactory
from twisted.python.failure import Failure
def _get_default_clock() -> IReactorTime:
    """
    Return the default reactor.

    This is a function so it can be monkey-patched in tests, specifically
    L{twisted.web.test.test_agent}.
    """
    from twisted.internet import reactor
    return cast(IReactorTime, reactor)