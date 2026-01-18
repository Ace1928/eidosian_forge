import os
from twisted.internet.defer import Deferred
from twisted.internet.interfaces import IReadDescriptor
from twisted.internet.posixbase import PosixReactorBase, _Waker
from twisted.internet.protocol import ServerFactory
from twisted.python.runtime import platform
from twisted.trial.unittest import TestCase
from twisted.internet import reactor
from twisted.internet.tcp import Port
class TrivialReactor(PosixReactorBase):

    def __init__(self):
        self._readers = {}
        self._writers = {}
        PosixReactorBase.__init__(self)

    def addReader(self, reader):
        self._readers[reader] = True

    def removeReader(self, reader):
        del self._readers[reader]

    def addWriter(self, writer):
        self._writers[writer] = True

    def removeWriter(self, writer):
        del self._writers[writer]