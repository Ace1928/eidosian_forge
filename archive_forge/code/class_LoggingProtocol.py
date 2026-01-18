from typing import Dict, List, Optional
import attr
import incremental
from twisted.application import service
from twisted.internet import error, protocol, reactor as _reactor
from twisted.logger import Logger
from twisted.protocols import basic
from twisted.python import deprecate
class LoggingProtocol(protocol.ProcessProtocol):
    service = None
    name = None

    def connectionMade(self):
        self._output = LineLogger()
        self._output.tag = self.name
        self._output.stream = 'stdout'
        self._output.service = self.service
        self._outputEmpty = True
        self._error = LineLogger()
        self._error.tag = self.name
        self._error.stream = 'stderr'
        self._error.service = self.service
        self._errorEmpty = True
        self._output.makeConnection(transport)
        self._error.makeConnection(transport)

    def outReceived(self, data):
        self._output.dataReceived(data)
        self._outputEmpty = data[-1] == b'\n'

    def errReceived(self, data):
        self._error.dataReceived(data)
        self._errorEmpty = data[-1] == b'\n'

    def processEnded(self, reason):
        if not self._outputEmpty:
            self._output.dataReceived(b'\n')
        if not self._errorEmpty:
            self._error.dataReceived(b'\n')
        self.service.connectionLost(self.name)

    @property
    def output(self):
        return self._output

    @property
    def empty(self):
        return self._outputEmpty