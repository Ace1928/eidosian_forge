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
class NewStyleTests(unittest.SynchronousTestCase):

    def setUp(self):
        """
        Create a pb server using L{Echoer} protocol and connect a client to it.
        """
        self.serverFactory = pb.PBServerFactory(Echoer())
        clientFactory = pb.PBClientFactory()
        client, self.server, self.pump = connectServerAndClient(test=self, clientFactory=clientFactory, serverFactory=self.serverFactory)
        self.ref = self.successResultOf(clientFactory.getRootObject())

    def tearDown(self):
        """
        Close client and server connections, reset values of L{NewStyleCopy2}
        class variables.
        """
        NewStyleCopy2.allocated = 0
        NewStyleCopy2.initialized = 0
        NewStyleCopy2.value = 1

    def test_newStyle(self):
        """
        Create a new style object, send it over the wire, and check the result.
        """
        orig = NewStyleCopy('value')
        d = self.ref.callRemote('echo', orig)
        self.pump.flush()

        def cb(res):
            self.assertIsInstance(res, NewStyleCopy)
            self.assertEqual(res.s, 'value')
            self.assertFalse(res is orig)
        d.addCallback(cb)
        return d

    def test_alloc(self):
        """
        Send a new style object and check the number of allocations.
        """
        orig = NewStyleCopy2()
        self.assertEqual(NewStyleCopy2.allocated, 1)
        self.assertEqual(NewStyleCopy2.initialized, 1)
        d = self.ref.callRemote('echo', orig)
        self.pump.flush()

        def cb(res):
            self.assertIsInstance(res, NewStyleCopy2)
            self.assertEqual(res.value, 2)
            self.assertEqual(NewStyleCopy2.allocated, 3)
            self.assertEqual(NewStyleCopy2.initialized, 1)
            self.assertIsNot(res, orig)
        d.addCallback(cb)
        return d

    def test_newStyleWithKeywords(self):
        """
        Create a new style object with keywords,
        send it over the wire, and check the result.
        """
        orig = NewStyleCopy('value1')
        d = self.ref.callRemote('echoWithKeywords', orig, keyword1='one', keyword2='two')
        self.pump.flush()

        def cb(res):
            self.assertIsInstance(res, tuple)
            self.assertIsInstance(res[0], NewStyleCopy)
            self.assertIsInstance(res[1], dict)
            self.assertEqual(res[0].s, 'value1')
            self.assertIsNot(res[0], orig)
            self.assertEqual(res[1], {'keyword1': 'one', 'keyword2': 'two'})
        d.addCallback(cb)
        return d