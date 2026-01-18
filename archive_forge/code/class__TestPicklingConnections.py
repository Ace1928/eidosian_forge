import unittest
import unittest.mock
import queue as pyqueue
import textwrap
import time
import io
import itertools
import sys
import os
import gc
import errno
import signal
import array
import socket
import random
import logging
import subprocess
import struct
import operator
import pickle #XXX: use dill?
import weakref
import warnings
import test.support
import test.support.script_helper
from test import support
from test.support import hashlib_helper
from test.support import import_helper
from test.support import os_helper
from test.support import socket_helper
from test.support import threading_helper
from test.support import warnings_helper
import_helper.import_module('multiprocess.synchronize')
import threading
import multiprocess as multiprocessing
import multiprocess.connection
import multiprocess.dummy
import multiprocess.heap
import multiprocess.managers
import multiprocess.pool
import multiprocess.queues
from multiprocess import util
from multiprocess.connection import wait
from multiprocess.managers import BaseManager, BaseProxy, RemoteError
@unittest.skipUnless(HAS_REDUCTION, 'test needs multiprocessing.reduction')
@hashlib_helper.requires_hashdigest('md5')
class _TestPicklingConnections(BaseTestCase):
    ALLOWED_TYPES = ('processes',)

    @classmethod
    def tearDownClass(cls):
        from multiprocess import resource_sharer
        resource_sharer.stop(timeout=support.LONG_TIMEOUT)

    @classmethod
    def _listener(cls, conn, families):
        for fam in families:
            l = cls.connection.Listener(family=fam)
            conn.send(l.address)
            new_conn = l.accept()
            conn.send(new_conn)
            new_conn.close()
            l.close()
        l = socket.create_server((socket_helper.HOST, 0))
        conn.send(l.getsockname())
        new_conn, addr = l.accept()
        conn.send(new_conn)
        new_conn.close()
        l.close()
        conn.recv()

    @classmethod
    def _remote(cls, conn):
        for address, msg in iter(conn.recv, None):
            client = cls.connection.Client(address)
            client.send(msg.upper())
            client.close()
        address, msg = conn.recv()
        client = socket.socket()
        client.connect(address)
        client.sendall(msg.upper())
        client.close()
        conn.close()

    def test_pickling(self):
        families = self.connection.families
        lconn, lconn0 = self.Pipe()
        lp = self.Process(target=self._listener, args=(lconn0, families))
        lp.daemon = True
        lp.start()
        lconn0.close()
        rconn, rconn0 = self.Pipe()
        rp = self.Process(target=self._remote, args=(rconn0,))
        rp.daemon = True
        rp.start()
        rconn0.close()
        for fam in families:
            msg = ('This connection uses family %s' % fam).encode('ascii')
            address = lconn.recv()
            rconn.send((address, msg))
            new_conn = lconn.recv()
            self.assertEqual(new_conn.recv(), msg.upper())
        rconn.send(None)
        msg = latin('This connection uses a normal socket')
        address = lconn.recv()
        rconn.send((address, msg))
        new_conn = lconn.recv()
        buf = []
        while True:
            s = new_conn.recv(100)
            if not s:
                break
            buf.append(s)
        buf = b''.join(buf)
        self.assertEqual(buf, msg.upper())
        new_conn.close()
        lconn.send(None)
        rconn.close()
        lconn.close()
        lp.join()
        rp.join()

    @classmethod
    def child_access(cls, conn):
        w = conn.recv()
        w.send('all is well')
        w.close()
        r = conn.recv()
        msg = r.recv()
        conn.send(msg * 2)
        conn.close()

    def test_access(self):
        conn, child_conn = self.Pipe()
        p = self.Process(target=self.child_access, args=(child_conn,))
        p.daemon = True
        p.start()
        child_conn.close()
        r, w = self.Pipe(duplex=False)
        conn.send(w)
        w.close()
        self.assertEqual(r.recv(), 'all is well')
        r.close()
        r, w = self.Pipe(duplex=False)
        conn.send(r)
        r.close()
        w.send('foobar')
        w.close()
        self.assertEqual(conn.recv(), 'foobar' * 2)
        p.join()