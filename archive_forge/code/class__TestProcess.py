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
class _TestProcess(BaseTestCase):
    ALLOWED_TYPES = ('processes', 'threads')

    def test_current(self):
        if self.TYPE == 'threads':
            self.skipTest('test not appropriate for {}'.format(self.TYPE))
        current = self.current_process()
        authkey = current.authkey
        self.assertTrue(current.is_alive())
        self.assertTrue(not current.daemon)
        self.assertIsInstance(authkey, bytes)
        self.assertTrue(len(authkey) > 0)
        self.assertEqual(current.ident, os.getpid())
        self.assertEqual(current.exitcode, None)

    def test_daemon_argument(self):
        if self.TYPE == 'threads':
            self.skipTest('test not appropriate for {}'.format(self.TYPE))
        proc0 = self.Process(target=self._test)
        self.assertEqual(proc0.daemon, self.current_process().daemon)
        proc1 = self.Process(target=self._test, daemon=True)
        self.assertTrue(proc1.daemon)
        proc2 = self.Process(target=self._test, daemon=False)
        self.assertFalse(proc2.daemon)

    @classmethod
    def _test(cls, q, *args, **kwds):
        current = cls.current_process()
        q.put(args)
        q.put(kwds)
        q.put(current.name)
        if cls.TYPE != 'threads':
            q.put(bytes(current.authkey))
            q.put(current.pid)

    def test_parent_process_attributes(self):
        if self.TYPE == 'threads':
            self.skipTest('test not appropriate for {}'.format(self.TYPE))
        self.assertIsNone(self.parent_process())
        rconn, wconn = self.Pipe(duplex=False)
        p = self.Process(target=self._test_send_parent_process, args=(wconn,))
        p.start()
        p.join()
        parent_pid, parent_name = rconn.recv()
        self.assertEqual(parent_pid, self.current_process().pid)
        self.assertEqual(parent_pid, os.getpid())
        self.assertEqual(parent_name, self.current_process().name)

    @classmethod
    def _test_send_parent_process(cls, wconn):
        from multiprocess.process import parent_process
        wconn.send([parent_process().pid, parent_process().name])

    def _test_parent_process(self):
        if self.TYPE == 'threads':
            self.skipTest('test not appropriate for {}'.format(self.TYPE))
        rconn, wconn = self.Pipe(duplex=False)
        p = self.Process(target=self._test_create_grandchild_process, args=(wconn,))
        p.start()
        if not rconn.poll(timeout=support.LONG_TIMEOUT):
            raise AssertionError('Could not communicate with child process')
        parent_process_status = rconn.recv()
        self.assertEqual(parent_process_status, 'alive')
        p.terminate()
        p.join()
        if not rconn.poll(timeout=support.LONG_TIMEOUT):
            raise AssertionError('Could not communicate with child process')
        parent_process_status = rconn.recv()
        self.assertEqual(parent_process_status, 'not alive')

    @classmethod
    def _test_create_grandchild_process(cls, wconn):
        p = cls.Process(target=cls._test_report_parent_status, args=(wconn,))
        p.start()
        time.sleep(300)

    @classmethod
    def _test_report_parent_status(cls, wconn):
        from multiprocess.process import parent_process
        wconn.send('alive' if parent_process().is_alive() else 'not alive')
        parent_process().join(timeout=support.SHORT_TIMEOUT)
        wconn.send('alive' if parent_process().is_alive() else 'not alive')

    def test_process(self):
        q = self.Queue(1)
        e = self.Event()
        args = (q, 1, 2)
        kwargs = {'hello': 23, 'bye': 2.54}
        name = 'SomeProcess'
        p = self.Process(target=self._test, args=args, kwargs=kwargs, name=name)
        p.daemon = True
        current = self.current_process()
        if self.TYPE != 'threads':
            self.assertEqual(p.authkey, current.authkey)
        self.assertEqual(p.is_alive(), False)
        self.assertEqual(p.daemon, True)
        self.assertNotIn(p, self.active_children())
        self.assertTrue(type(self.active_children()) is list)
        self.assertEqual(p.exitcode, None)
        p.start()
        self.assertEqual(p.exitcode, None)
        self.assertEqual(p.is_alive(), True)
        self.assertIn(p, self.active_children())
        self.assertEqual(q.get(), args[1:])
        self.assertEqual(q.get(), kwargs)
        self.assertEqual(q.get(), p.name)
        if self.TYPE != 'threads':
            self.assertEqual(q.get(), current.authkey)
            self.assertEqual(q.get(), p.pid)
        p.join()
        self.assertEqual(p.exitcode, 0)
        self.assertEqual(p.is_alive(), False)
        self.assertNotIn(p, self.active_children())
        close_queue(q)

    @unittest.skipUnless(threading._HAVE_THREAD_NATIVE_ID, 'needs native_id')
    def test_process_mainthread_native_id(self):
        if self.TYPE == 'threads':
            self.skipTest('test not appropriate for {}'.format(self.TYPE))
        current_mainthread_native_id = threading.main_thread().native_id
        q = self.Queue(1)
        p = self.Process(target=self._test_process_mainthread_native_id, args=(q,))
        p.start()
        child_mainthread_native_id = q.get()
        p.join()
        close_queue(q)
        self.assertNotEqual(current_mainthread_native_id, child_mainthread_native_id)

    @classmethod
    def _test_process_mainthread_native_id(cls, q):
        mainthread_native_id = threading.main_thread().native_id
        q.put(mainthread_native_id)

    @classmethod
    def _sleep_some(cls):
        time.sleep(100)

    @classmethod
    def _test_sleep(cls, delay):
        time.sleep(delay)

    def _kill_process(self, meth):
        if self.TYPE == 'threads':
            self.skipTest('test not appropriate for {}'.format(self.TYPE))
        p = self.Process(target=self._sleep_some)
        p.daemon = True
        p.start()
        self.assertEqual(p.is_alive(), True)
        self.assertIn(p, self.active_children())
        self.assertEqual(p.exitcode, None)
        join = TimingWrapper(p.join)
        self.assertEqual(join(0), None)
        self.assertTimingAlmostEqual(join.elapsed, 0.0)
        self.assertEqual(p.is_alive(), True)
        self.assertEqual(join(-1), None)
        self.assertTimingAlmostEqual(join.elapsed, 0.0)
        self.assertEqual(p.is_alive(), True)
        time.sleep(1)
        meth(p)
        if hasattr(signal, 'alarm'):

            def handler(*args):
                raise RuntimeError('join took too long: %s' % p)
            old_handler = signal.signal(signal.SIGALRM, handler)
            try:
                signal.alarm(10)
                self.assertEqual(join(), None)
            finally:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)
        else:
            self.assertEqual(join(), None)
        self.assertTimingAlmostEqual(join.elapsed, 0.0)
        self.assertEqual(p.is_alive(), False)
        self.assertNotIn(p, self.active_children())
        p.join()
        return p.exitcode

    def test_terminate(self):
        exitcode = self._kill_process(multiprocessing.Process.terminate)
        if os.name != 'nt':
            self.assertEqual(exitcode, -signal.SIGTERM)

    def test_kill(self):
        exitcode = self._kill_process(multiprocessing.Process.kill)
        if os.name != 'nt':
            self.assertEqual(exitcode, -signal.SIGKILL)

    def test_cpu_count(self):
        try:
            cpus = multiprocessing.cpu_count()
        except NotImplementedError:
            cpus = 1
        self.assertTrue(type(cpus) is int)
        self.assertTrue(cpus >= 1)

    def test_active_children(self):
        self.assertEqual(type(self.active_children()), list)
        p = self.Process(target=time.sleep, args=(DELTA,))
        self.assertNotIn(p, self.active_children())
        p.daemon = True
        p.start()
        self.assertIn(p, self.active_children())
        p.join()
        self.assertNotIn(p, self.active_children())

    @classmethod
    def _test_recursion(cls, wconn, id):
        wconn.send(id)
        if len(id) < 2:
            for i in range(2):
                p = cls.Process(target=cls._test_recursion, args=(wconn, id + [i]))
                p.start()
                p.join()

    @unittest.skipIf(True, 'fails with is_dill(obj, child=True)')
    def test_recursion(self):
        rconn, wconn = self.Pipe(duplex=False)
        self._test_recursion(wconn, [])
        time.sleep(DELTA)
        result = []
        while rconn.poll():
            result.append(rconn.recv())
        expected = [[], [0], [0, 0], [0, 1], [1], [1, 0], [1, 1]]
        self.assertEqual(result, expected)

    @classmethod
    def _test_sentinel(cls, event):
        event.wait(10.0)

    def test_sentinel(self):
        if self.TYPE == 'threads':
            self.skipTest('test not appropriate for {}'.format(self.TYPE))
        event = self.Event()
        p = self.Process(target=self._test_sentinel, args=(event,))
        with self.assertRaises(ValueError):
            p.sentinel
        p.start()
        self.addCleanup(p.join)
        sentinel = p.sentinel
        self.assertIsInstance(sentinel, int)
        self.assertFalse(wait_for_handle(sentinel, timeout=0.0))
        event.set()
        p.join()
        self.assertTrue(wait_for_handle(sentinel, timeout=1))

    @classmethod
    def _test_close(cls, rc=0, q=None):
        if q is not None:
            q.get()
        sys.exit(rc)

    def test_close(self):
        if self.TYPE == 'threads':
            self.skipTest('test not appropriate for {}'.format(self.TYPE))
        q = self.Queue()
        p = self.Process(target=self._test_close, kwargs={'q': q})
        p.daemon = True
        p.start()
        self.assertEqual(p.is_alive(), True)
        with self.assertRaises(ValueError):
            p.close()
        q.put(None)
        p.join()
        self.assertEqual(p.is_alive(), False)
        self.assertEqual(p.exitcode, 0)
        p.close()
        with self.assertRaises(ValueError):
            p.is_alive()
        with self.assertRaises(ValueError):
            p.join()
        with self.assertRaises(ValueError):
            p.terminate()
        p.close()
        wr = weakref.ref(p)
        del p
        gc.collect()
        self.assertIs(wr(), None)
        close_queue(q)

    def test_many_processes(self):
        if self.TYPE == 'threads':
            self.skipTest('test not appropriate for {}'.format(self.TYPE))
        sm = multiprocessing.get_start_method()
        travis = os.environ.get('COVERAGE')
        N = (1 if travis else 5) if sm == 'spawn' else 100
        procs = [self.Process(target=self._test_sleep, args=(0.01,)) for i in range(N)]
        for p in procs:
            p.start()
        for p in procs:
            join_process(p)
        for p in procs:
            self.assertEqual(p.exitcode, 0)
        procs = [self.Process(target=self._sleep_some) for i in range(N)]
        for p in procs:
            p.start()
        time.sleep(0.001)
        for p in procs:
            p.terminate()
        for p in procs:
            join_process(p)
        if os.name != 'nt':
            exitcodes = [-signal.SIGTERM]
            if sys.platform == 'darwin':
                exitcodes.append(-signal.SIGKILL)
            for p in procs:
                self.assertIn(p.exitcode, exitcodes)

    def test_lose_target_ref(self):
        c = DummyCallable()
        wr = weakref.ref(c)
        q = self.Queue()
        p = self.Process(target=c, args=(q, c))
        del c
        p.start()
        p.join()
        gc.collect()
        self.assertIs(wr(), None)
        self.assertEqual(q.get(), 5)
        close_queue(q)

    @classmethod
    def _test_child_fd_inflation(self, evt, q):
        q.put(os_helper.fd_count())
        evt.wait()

    def test_child_fd_inflation(self):
        if self.TYPE == 'threads':
            self.skipTest('test not appropriate for {}'.format(self.TYPE))
        sm = multiprocessing.get_start_method()
        if sm == 'fork':
            self.skipTest('test not appropriate for {}'.format(sm))
        N = 5
        evt = self.Event()
        q = self.Queue()
        procs = [self.Process(target=self._test_child_fd_inflation, args=(evt, q)) for i in range(N)]
        for p in procs:
            p.start()
        try:
            fd_counts = [q.get() for i in range(N)]
            self.assertEqual(len(set(fd_counts)), 1, fd_counts)
        finally:
            evt.set()
            for p in procs:
                p.join()
            close_queue(q)

    @classmethod
    def _test_wait_for_threads(self, evt):

        def func1():
            time.sleep(0.5)
            evt.set()

        def func2():
            time.sleep(20)
            evt.clear()
        threading.Thread(target=func1).start()
        threading.Thread(target=func2, daemon=True).start()

    def test_wait_for_threads(self):
        if self.TYPE == 'threads':
            self.skipTest('test not appropriate for {}'.format(self.TYPE))
        evt = self.Event()
        proc = self.Process(target=self._test_wait_for_threads, args=(evt,))
        proc.start()
        proc.join()
        self.assertTrue(evt.is_set())

    @classmethod
    def _test_error_on_stdio_flush(self, evt, break_std_streams={}):
        for stream_name, action in break_std_streams.items():
            if action == 'close':
                stream = io.StringIO()
                stream.close()
            else:
                assert action == 'remove'
                stream = None
            setattr(sys, stream_name, None)
        evt.set()

    def test_error_on_stdio_flush_1(self):
        streams = [io.StringIO(), None]
        streams[0].close()
        for stream_name in ('stdout', 'stderr'):
            for stream in streams:
                old_stream = getattr(sys, stream_name)
                setattr(sys, stream_name, stream)
                try:
                    evt = self.Event()
                    proc = self.Process(target=self._test_error_on_stdio_flush, args=(evt,))
                    proc.start()
                    proc.join()
                    self.assertTrue(evt.is_set())
                    self.assertEqual(proc.exitcode, 0)
                finally:
                    setattr(sys, stream_name, old_stream)

    def test_error_on_stdio_flush_2(self):
        for stream_name in ('stdout', 'stderr'):
            for action in ('close', 'remove'):
                old_stream = getattr(sys, stream_name)
                try:
                    evt = self.Event()
                    proc = self.Process(target=self._test_error_on_stdio_flush, args=(evt, {stream_name: action}))
                    proc.start()
                    proc.join()
                    self.assertTrue(evt.is_set())
                    self.assertEqual(proc.exitcode, 0)
                finally:
                    setattr(sys, stream_name, old_stream)

    @classmethod
    def _sleep_and_set_event(self, evt, delay=0.0):
        time.sleep(delay)
        evt.set()

    def check_forkserver_death(self, signum):
        if self.TYPE == 'threads':
            self.skipTest('test not appropriate for {}'.format(self.TYPE))
        sm = multiprocessing.get_start_method()
        if sm != 'forkserver':
            self.skipTest('test not appropriate for {}'.format(sm))
        from multiprocess.forkserver import _forkserver
        _forkserver.ensure_running()
        delay = 0.5
        evt = self.Event()
        proc = self.Process(target=self._sleep_and_set_event, args=(evt, delay))
        proc.start()
        pid = _forkserver._forkserver_pid
        os.kill(pid, signum)
        time.sleep(delay * 2.0)
        evt2 = self.Event()
        proc2 = self.Process(target=self._sleep_and_set_event, args=(evt2,))
        proc2.start()
        proc2.join()
        self.assertTrue(evt2.is_set())
        self.assertEqual(proc2.exitcode, 0)
        proc.join()
        self.assertTrue(evt.is_set())
        self.assertIn(proc.exitcode, (0, 255))

    def test_forkserver_sigint(self):
        self.check_forkserver_death(signal.SIGINT)

    def test_forkserver_sigkill(self):
        if os.name != 'nt':
            self.check_forkserver_death(signal.SIGKILL)