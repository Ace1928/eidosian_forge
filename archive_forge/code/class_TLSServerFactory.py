import inspect
import sys
from typing import List
from unittest import skipIf
from zope.interface import directlyProvides
import twisted.mail._pop3client
from twisted.internet import defer, error, interfaces, protocol, reactor
from twisted.internet.testing import StringTransport
from twisted.mail.pop3 import (
from twisted.mail.test import pop3testserver
from twisted.protocols import basic, loopback
from twisted.python import log
from twisted.trial.unittest import TestCase
class TLSServerFactory(protocol.ServerFactory):

    class protocol(basic.LineReceiver):
        context = None
        output: List[bytes] = []

        def connectionMade(self):
            self.factory.input = []
            self.output = self.output[:]
            for line in self.output.pop(0):
                self.sendLine(line)

        def lineReceived(self, line):
            self.factory.input.append(line)
            [self.sendLine(l) for l in self.output.pop(0)]
            if line == b'STLS':
                self.transport.startTLS(self.context)