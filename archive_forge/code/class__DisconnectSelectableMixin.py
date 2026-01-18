import socket
import sys
from typing import Sequence
from zope.interface import classImplements, implementer
from twisted.internet import error, tcp, udp
from twisted.internet.base import ReactorBase
from twisted.internet.interfaces import (
from twisted.internet.main import CONNECTION_DONE, CONNECTION_LOST
from twisted.python import failure, log
from twisted.python.runtime import platform, platformType
from ._signals import (
class _DisconnectSelectableMixin:
    """
    Mixin providing the C{_disconnectSelectable} method.
    """

    def _disconnectSelectable(self, selectable, why, isRead, faildict={error.ConnectionDone: failure.Failure(error.ConnectionDone()), error.ConnectionLost: failure.Failure(error.ConnectionLost())}):
        """
        Utility function for disconnecting a selectable.

        Supports half-close notification, isRead should be boolean indicating
        whether error resulted from doRead().
        """
        self.removeReader(selectable)
        f = faildict.get(why.__class__)
        if f:
            if isRead and why.__class__ == error.ConnectionDone and IHalfCloseableDescriptor.providedBy(selectable):
                selectable.readConnectionLost(f)
            else:
                self.removeWriter(selectable)
                selectable.connectionLost(f)
        else:
            self.removeWriter(selectable)
            selectable.connectionLost(failure.Failure(why))