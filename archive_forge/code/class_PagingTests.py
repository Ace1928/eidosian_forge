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
class PagingTests(unittest.TestCase):
    """
    Test pb objects sending data by pages.
    """

    def setUp(self):
        """
        Create a file used to test L{util.FilePager}.
        """
        self.filename = self.mktemp()
        with open(self.filename, 'wb') as f:
            f.write(bigString)

    def test_pagingWithCallback(self):
        """
        Test L{util.StringPager}, passing a callback to fire when all pages
        are sent.
        """
        c, s, pump = connectedServerAndClient(test=self)
        s.setNameForLocal('foo', Pagerizer(finishedCallback, 'hello', value=10))
        x = c.remoteForName('foo')
        l = []
        util.getAllPages(x, 'getPages').addCallback(l.append)
        while not l:
            pump.pump()
        self.assertEqual(b''.join(l[0]), bigString, 'Pages received not equal to pages sent!')
        self.assertEqual(callbackArgs, ('hello',), 'Completed callback not invoked')
        self.assertEqual(callbackKeyword, {'value': 10}, 'Completed callback not invoked')

    def test_pagingWithoutCallback(self):
        """
        Test L{util.StringPager} without a callback.
        """
        c, s, pump = connectedServerAndClient(test=self)
        s.setNameForLocal('foo', Pagerizer(None))
        x = c.remoteForName('foo')
        l = []
        util.getAllPages(x, 'getPages').addCallback(l.append)
        while not l:
            pump.pump()
        self.assertEqual(b''.join(l[0]), bigString, 'Pages received not equal to pages sent!')

    def test_emptyFilePaging(self):
        """
        Test L{util.FilePager}, sending an empty file.
        """
        filenameEmpty = self.mktemp()
        open(filenameEmpty, 'w').close()
        c, s, pump = connectedServerAndClient(test=self)
        pagerizer = FilePagerizer(filenameEmpty, None)
        s.setNameForLocal('bar', pagerizer)
        x = c.remoteForName('bar')
        l = []
        util.getAllPages(x, 'getPages').addCallback(l.append)
        ttl = 10
        while not l and ttl > 0:
            pump.pump()
            ttl -= 1
        if not ttl:
            self.fail('getAllPages timed out')
        self.assertEqual(b''.join(l[0]), b'', 'Pages received not equal to pages sent!')

    def test_filePagingWithCallback(self):
        """
        Test L{util.FilePager}, passing a callback to fire when all pages
        are sent, and verify that the pager doesn't keep chunks in memory.
        """
        c, s, pump = connectedServerAndClient(test=self)
        pagerizer = FilePagerizer(self.filename, finishedCallback, 'frodo', value=9)
        s.setNameForLocal('bar', pagerizer)
        x = c.remoteForName('bar')
        l = []
        util.getAllPages(x, 'getPages').addCallback(l.append)
        while not l:
            pump.pump()
        self.assertEqual(b''.join(l[0]), bigString, 'Pages received not equal to pages sent!')
        self.assertEqual(callbackArgs, ('frodo',), 'Completed callback not invoked')
        self.assertEqual(callbackKeyword, {'value': 9}, 'Completed callback not invoked')
        self.assertEqual(pagerizer.pager.chunks, [])

    def test_filePagingWithoutCallback(self):
        """
        Test L{util.FilePager} without a callback.
        """
        c, s, pump = connectedServerAndClient(test=self)
        pagerizer = FilePagerizer(self.filename, None)
        s.setNameForLocal('bar', pagerizer)
        x = c.remoteForName('bar')
        l = []
        util.getAllPages(x, 'getPages').addCallback(l.append)
        while not l:
            pump.pump()
        self.assertEqual(b''.join(l[0]), bigString, 'Pages received not equal to pages sent!')
        self.assertEqual(pagerizer.pager.chunks, [])