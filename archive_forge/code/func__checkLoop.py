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
def _checkLoop(self):
    """
        Start or stop a C{LoopingCall} based on whether there are readers and
        writers.
        """
    if self._readers or self._writers:
        if self._loop is None:
            from twisted.internet.task import _EPSILON, LoopingCall
            self._loop = LoopingCall(self.iterate)
            self._loop.clock = self._reactor
            self._loop.start(_EPSILON, now=False)
    elif self._loop:
        self._loop.stop()
        self._loop = None