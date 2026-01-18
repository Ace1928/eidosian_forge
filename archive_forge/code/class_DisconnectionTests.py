import gc
import os
import sys
import time
import weakref
from collections import deque
from io import BytesIO as StringIO
from typing import Dict
from zope.interface import Interface, implementer
from twisted.cred import checkers, credentials, portal
from twisted.cred.error import UnauthorizedLogin, UnhandledCredentials
from twisted.internet import address, main, protocol, reactor
from twisted.internet.defer import Deferred, gatherResults, succeed
from twisted.internet.error import ConnectionRefusedError
from twisted.internet.testing import _FakeConnector
from twisted.protocols.policies import WrappingFactory
from twisted.python import failure, log
from twisted.python.compat import iterbytes
from twisted.spread import jelly, pb, publish, util
from twisted.trial import unittest
class DisconnectionTests(unittest.TestCase):
    """
    Test disconnection callbacks.
    """

    def error(self, *args):
        raise RuntimeError(f"I shouldn't have been called: {args}")

    def gotDisconnected(self):
        """
        Called on broker disconnect.
        """
        self.gotCallback = 1

    def objectDisconnected(self, o):
        """
        Called on RemoteReference disconnect.
        """
        self.assertEqual(o, self.remoteObject)
        self.objectCallback = 1

    def test_badSerialization(self):
        c, s, pump = connectedServerAndClient(test=self)
        pump.pump()
        s.setNameForLocal('o', BadCopySet())
        g = c.remoteForName('o')
        l = []
        g.callRemote('setBadCopy', BadCopyable()).addErrback(l.append)
        pump.flush()
        self.assertEqual(len(l), 1)

    def test_disconnection(self):
        c, s, pump = connectedServerAndClient(test=self)
        pump.pump()
        s.setNameForLocal('o', SimpleRemote())
        r = c.remoteForName('o')
        pump.pump()
        pump.pump()
        pump.pump()
        c.notifyOnDisconnect(self.error)
        self.assertIn(self.error, c.disconnects)
        c.dontNotifyOnDisconnect(self.error)
        self.assertNotIn(self.error, c.disconnects)
        r.notifyOnDisconnect(self.error)
        self.assertIn(r._disconnected, c.disconnects)
        self.assertIn(self.error, r.disconnectCallbacks)
        r.dontNotifyOnDisconnect(self.error)
        self.assertNotIn(r._disconnected, c.disconnects)
        self.assertNotIn(self.error, r.disconnectCallbacks)
        c.notifyOnDisconnect(self.gotDisconnected)
        r.notifyOnDisconnect(self.objectDisconnected)
        self.remoteObject = r
        c.connectionLost(failure.Failure(main.CONNECTION_DONE))
        self.assertTrue(self.gotCallback)
        self.assertTrue(self.objectCallback)