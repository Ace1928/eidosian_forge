from __future__ import absolute_import
import threading
import warnings
import subprocess
import sys
from unittest import SkipTest, TestCase
import twisted
from twisted.python.log import PythonLoggingObserver
from twisted.python import log
from twisted.python.runtime import platform
from twisted.internet.task import Clock
from .._eventloop import EventLoop, ThreadLogObserver, _store
from ..tests import crochet_directory
import sys
import crochet
import sys
from logging import StreamHandler, Formatter, getLogger, DEBUG
import crochet
from twisted.python import log
from twisted.logger import Logger
import time
class SetupTests(TestCase):
    """
    Tests for setup().
    """

    def test_first_runs_reactor(self):
        """
        With it first call, setup() runs the reactor in a thread.
        """
        reactor = FakeReactor()
        EventLoop(lambda: reactor, lambda f, *g: None).setup()
        reactor.started.wait(5)
        self.assertNotEqual(reactor.thread_id, None)
        self.assertNotEqual(reactor.thread_id, threading.current_thread().ident)
        self.assertFalse(reactor.installSignalHandlers)

    def test_second_does_nothing(self):
        """
        The second call to setup() does nothing.
        """
        reactor = FakeReactor()
        s = EventLoop(lambda: reactor, lambda f, *g: None)
        s.setup()
        s.setup()
        reactor.started.wait(5)
        self.assertEqual(reactor.runs, 1)

    def test_stop_on_exit(self):
        """
        setup() registers an exit handler that stops the reactor, and an exit
        handler that logs stashed EventualResults.
        """
        atexit = []
        reactor = FakeReactor()
        s = EventLoop(lambda: reactor, lambda f, *args: atexit.append((f, args)))
        s.setup()
        self.assertEqual(len(atexit), 2)
        self.assertFalse(reactor.stopping)
        f, args = atexit[0]
        self.assertEqual(f, reactor.callFromThread)
        self.assertEqual(args, (reactor.stop,))
        f(*args)
        self.assertTrue(reactor.stopping)
        f, args = atexit[1]
        self.assertEqual(f, _store.log_errors)
        self.assertEqual(args, ())
        f(*args)

    def test_runs_with_lock(self):
        """
        All code in setup() and no_setup() is protected by a lock.
        """
        self.assertTrue(EventLoop.setup.synchronized)
        self.assertTrue(EventLoop.no_setup.synchronized)

    def test_logging(self):
        """
        setup() registers a PythonLoggingObserver wrapped in a
        ThreadLogObserver, removing the default log observer.
        """
        logging = []

        def fakeStartLoggingWithObserver(observer, setStdout=1):
            self.assertIsInstance(observer, ThreadLogObserver)
            wrapped = observer._observer
            expected = PythonLoggingObserver.emit
            expected = getattr(expected, '__func__', expected)
            self.assertIs(wrapped.__func__, expected)
            self.assertEqual(setStdout, False)
            self.assertTrue(reactor.in_call_from_thread)
            logging.append(observer)
        reactor = FakeReactor()
        loop = EventLoop(lambda: reactor, lambda f, *g: None, fakeStartLoggingWithObserver)
        loop.setup()
        self.assertTrue(logging)
        logging[0].stop()

    def test_stop_logging_on_exit(self):
        """
        setup() registers a reactor shutdown event that stops the logging
        thread.
        """
        observers = []
        reactor = FakeReactor()
        s = EventLoop(lambda: reactor, lambda f, *arg: None, lambda observer, setStdout=1: observers.append(observer))
        s.setup()
        self.addCleanup(observers[0].stop)
        self.assertIn(('after', 'shutdown', observers[0].stop), reactor.events)

    def test_warnings_untouched(self):
        """
        setup() ensure the warnings module's showwarning is unmodified,
        overriding the change made by normal Twisted logging setup.
        """

        def fakeStartLoggingWithObserver(observer, setStdout=1):
            warnings.showwarning = log.showwarning
            self.addCleanup(observer.stop)
        original = warnings.showwarning
        reactor = FakeReactor()
        loop = EventLoop(lambda: reactor, lambda f, *g: None, fakeStartLoggingWithObserver)
        loop.setup()
        self.assertIs(warnings.showwarning, original)

    def test_start_watchdog_thread(self):
        """
        setup() starts the shutdown watchdog thread.
        """
        thread = FakeThread()
        reactor = FakeReactor()
        loop = EventLoop(lambda: reactor, lambda *args: None, watchdog_thread=thread)
        loop.setup()
        self.assertTrue(thread.started)

    def test_no_setup(self):
        """
        If called first, no_setup() makes subsequent calls to setup() do
        nothing.
        """
        observers = []
        atexit = []
        thread = FakeThread()
        reactor = FakeReactor()
        loop = EventLoop(lambda: reactor, lambda f, *arg: atexit.append(f), lambda observer, *a, **kw: observers.append(observer), watchdog_thread=thread)
        loop.no_setup()
        loop.setup()
        self.assertFalse(observers)
        self.assertFalse(atexit)
        self.assertFalse(reactor.runs)
        self.assertFalse(thread.started)

    def test_no_setup_after_setup(self):
        """
        If called after setup(), no_setup() throws an exception.
        """
        reactor = FakeReactor()
        s = EventLoop(lambda: reactor, lambda f, *g: None)
        s.setup()
        self.assertRaises(RuntimeError, s.no_setup)

    def test_setup_registry_shutdown(self):
        """
        ResultRegistry.stop() is registered to run before reactor shutdown by
        setup().
        """
        reactor = FakeReactor()
        s = EventLoop(lambda: reactor, lambda f, *g: None)
        s.setup()
        self.assertEqual(reactor.events, [('before', 'shutdown', s._registry.stop)])

    def test_no_setup_registry_shutdown(self):
        """
        ResultRegistry.stop() is registered to run before reactor shutdown by
        setup().
        """
        reactor = FakeReactor()
        s = EventLoop(lambda: reactor, lambda f, *g: None)
        s.no_setup()
        self.assertEqual(reactor.events, [('before', 'shutdown', s._registry.stop)])