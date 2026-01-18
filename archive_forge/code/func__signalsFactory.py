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
def _signalsFactory(self) -> SignalHandling:
    """
        Customize reactor signal handling to support child processes on POSIX
        platforms.
        """
    baseHandling = super()._signalsFactory()
    if platformType == 'posix':
        return _MultiSignalHandling((baseHandling, _ChildSignalHandling(self._addInternalReader, self._removeInternalReader)))
    return baseHandling