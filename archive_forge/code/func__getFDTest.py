import os
import socket
import traceback
from unittest import skipIf
from zope.interface import implementer
from twisted.internet.abstract import FileDescriptor
from twisted.internet.interfaces import IReactorFDSet, IReadDescriptor
from twisted.internet.tcp import EINPROGRESS, EWOULDBLOCK
from twisted.internet.test.reactormixins import ReactorBuilder
from twisted.python.runtime import platform
from twisted.trial.unittest import SkipTest
def _getFDTest(self, kind):
    """
        Helper for getReaders and getWriters tests.
        """
    reactor = self.buildReactor()
    get = getattr(reactor, 'get' + kind + 's')
    add = getattr(reactor, 'add' + kind)
    remove = getattr(reactor, 'remove' + kind)
    client, server = self._connectedPair()
    self.assertNotIn(client, get())
    self.assertNotIn(server, get())
    add(client)
    self.assertIn(client, get())
    self.assertNotIn(server, get())
    remove(client)
    self.assertNotIn(client, get())
    self.assertNotIn(server, get())