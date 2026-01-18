import os
from twisted.internet.defer import Deferred
from twisted.internet.interfaces import IReadDescriptor
from twisted.internet.posixbase import PosixReactorBase, _Waker
from twisted.internet.protocol import ServerFactory
from twisted.python.runtime import platform
from twisted.trial.unittest import TestCase
from twisted.internet import reactor
from twisted.internet.tcp import Port
def addReader(self, reader: IReadDescriptor) -> None:
    """
        Ignore the reader.  This is necessary because the waker will be
        added.  However, we won't actually monitor it for any events.
        """