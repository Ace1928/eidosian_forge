from __future__ import annotations
from typing import (
from zope.interface import Attribute, Interface
from twisted.python.failure import Failure
class _ISupportsExitSignalCapturing(Interface):
    """
    An implementor of L{_ISupportsExitSignalCapturing} will capture the
    value of any delivered exit signal (SIGINT, SIGTERM, SIGBREAK) for which
    it has installed a handler.  The caught signal number is made available in
    the _exitSignal attribute.
    """
    _exitSignal = Attribute('\n        C{int} or C{None}, the integer exit signal delivered to the\n        application, or None if no signal was delivered.\n        ')