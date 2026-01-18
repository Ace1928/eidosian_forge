from __future__ import absolute_import
import threading
import subprocess
import time
import gc
import sys
import weakref
import tempfile
import os
import inspect
from unittest import SkipTest
from twisted.trial.unittest import TestCase
from twisted.internet.defer import succeed, Deferred, fail, CancelledError
from twisted.python.failure import Failure
from twisted.python import threadable
from twisted.python.runtime import platform
from .._eventloop import (
from .test_setup import FakeReactor
from .. import (
from ..tests import crochet_directory
import os, threading, signal, time, sys
import crochet
from twisted.internet.defer import Deferred
import os, threading, signal, time, sys
from twisted.internet.defer import Deferred
from twisted.internet import reactor
import crochet
import os, threading, signal, time, sys
from twisted.internet.defer import Deferred
from twisted.internet import reactor
import crochet
import os, threading, signal, time, sys
from twisted.internet.defer import Deferred, CancelledError
import crochet
from crochet import EventualResult
from twisted.internet.defer import Deferred
import os, threading, signal, time, sys
import crochet
from twisted.internet.defer import Deferred
import os, threading, signal, time, sys
from twisted.internet.defer import Deferred
from twisted.internet import reactor
import crochet
class EventualResultTests(TestCase):
    """
    Tests for EventualResult.
    """

    def setUp(self):
        self.patch(threadable, 'isInIOThread', lambda: False)

    def test_success_result(self):
        """
        wait() returns the value the Deferred fired with.
        """
        dr = EventualResult(succeed(123), None)
        self.assertEqual(dr.wait(0.1), 123)

    def test_later_success_result(self):
        """
        wait() returns the value the Deferred fired with, in the case where
        the Deferred is fired after wait() is called.
        """
        d = Deferred()
        dr = EventualResult(d, None)
        result_list = []
        done = append_in_thread(result_list, dr.wait, 100)
        time.sleep(0.1)
        d.callback(345)
        done.wait(100)
        self.assertEqual(result_list, [True, 345])

    def test_success_result_twice(self):
        """
        A second call to wait() returns same value as the first call.
        """
        dr = EventualResult(succeed(123), None)
        self.assertEqual(dr.wait(0.1), 123)
        self.assertEqual(dr.wait(0.1), 123)

    def test_failure_result(self):
        """
        wait() raises the exception the Deferred fired with.
        """
        dr = EventualResult(fail(RuntimeError()), None)
        self.assertRaises(RuntimeError, dr.wait, 0.1)

    def test_later_failure_result(self):
        """
        wait() raises the exception the Deferred fired with, in the case
        where the Deferred is fired after wait() is called.
        """
        d = Deferred()
        dr = EventualResult(d, None)
        result_list = []
        done = append_in_thread(result_list, dr.wait, 100)
        time.sleep(0.1)
        d.errback(RuntimeError())
        done.wait(100)
        self.assertEqual((result_list[0], result_list[1].__class__), (False, RuntimeError))

    def test_failure_result_twice(self):
        """
        A second call to wait() raises same value as the first call.
        """
        dr = EventualResult(fail(ZeroDivisionError()), None)
        self.assertRaises(ZeroDivisionError, dr.wait, 0.1)
        self.assertRaises(ZeroDivisionError, dr.wait, 0.1)

    def test_timeout(self):
        """
        If no result is available, wait(timeout) will throw a TimeoutError.
        """
        start = time.time()
        dr = EventualResult(Deferred(), None)
        self.assertRaises(TimeoutError, dr.wait, timeout=0.03)
        self.assertTrue(abs(time.time() - start) < 0.05)

    def test_timeout_twice(self):
        """
        If no result is available, a second call to wait(timeout) will also
        result in a TimeoutError exception.
        """
        dr = EventualResult(Deferred(), None)
        self.assertRaises(TimeoutError, dr.wait, timeout=0.01)
        self.assertRaises(TimeoutError, dr.wait, timeout=0.01)

    def test_timeout_then_result(self):
        """
        If a result becomes available after a timeout, a second call to
        wait() will return it.
        """
        d = Deferred()
        dr = EventualResult(d, None)
        self.assertRaises(TimeoutError, dr.wait, timeout=0.01)
        d.callback(u'value')
        self.assertEqual(dr.wait(0.1), u'value')
        self.assertEqual(dr.wait(0.1), u'value')

    def test_reactor_thread_disallowed(self):
        """
        wait() cannot be called from the reactor thread.
        """
        self.patch(threadable, 'isInIOThread', lambda: True)
        d = Deferred()
        dr = EventualResult(d, None)
        self.assertRaises(RuntimeError, dr.wait, 0)

    def test_cancel(self):
        """
        cancel() cancels the wrapped Deferred, running cancellation in the
        event loop thread.
        """
        reactor = FakeReactor()
        cancelled = []

        def error(f):
            cancelled.append(reactor.in_call_from_thread)
            cancelled.append(f)
        d = Deferred().addErrback(error)
        dr = EventualResult(d, _reactor=reactor)
        dr.cancel()
        self.assertTrue(cancelled[0])
        self.assertIsInstance(cancelled[1].value, CancelledError)

    def test_stash(self):
        """
        EventualResult.stash() stores the object in the global ResultStore.
        """
        dr = EventualResult(Deferred(), None)
        uid = dr.stash()
        self.assertIdentical(dr, _store.retrieve(uid))

    def test_original_failure(self):
        """
        original_failure() returns the underlying Failure of the Deferred
        wrapped by the EventualResult.
        """
        try:
            1 / 0
        except ZeroDivisionError:
            f = Failure()
        dr = EventualResult(fail(f), None)
        self.assertIdentical(dr.original_failure(), f)

    def test_original_failure_no_result(self):
        """
        If there is no result yet, original_failure() returns None.
        """
        dr = EventualResult(Deferred(), None)
        self.assertIdentical(dr.original_failure(), None)

    def test_original_failure_not_error(self):
        """
        If the result is not an error, original_failure() returns None.
        """
        dr = EventualResult(succeed(3), None)
        self.assertIdentical(dr.original_failure(), None)

    def test_error_logged_no_wait(self):
        """
        If the result is an error and wait() was never called, the error will
        be logged once the EventualResult is garbage-collected.
        """
        dr = EventualResult(fail(ZeroDivisionError()), None)
        del dr
        gc.collect()
        excs = self.flushLoggedErrors(ZeroDivisionError)
        self.assertEqual(len(excs), 1)

    def test_error_logged_wait_timeout(self):
        """
        If the result is an error and wait() was called but timed out, the
        error will be logged once the EventualResult is garbage-collected.
        """
        d = Deferred()
        dr = EventualResult(d, None)
        try:
            dr.wait(0)
        except TimeoutError:
            pass
        d.errback(ZeroDivisionError())
        del dr
        if sys.version_info[0] == 2:
            sys.exc_clear()
        gc.collect()
        excs = self.flushLoggedErrors(ZeroDivisionError)
        self.assertEqual(len(excs), 1)

    def test_error_after_gc_logged(self):
        """
        If the result is an error that occurs after all user references to the
        EventualResult are lost, the error is still logged.
        """
        d = Deferred()
        dr = EventualResult(d, None)
        del dr
        d.errback(ZeroDivisionError())
        gc.collect()
        excs = self.flushLoggedErrors(ZeroDivisionError)
        self.assertEqual(len(excs), 1)

    def test_control_c_is_possible(self):
        """
        If you're wait()ing on an EventualResult in main thread, make sure the
        KeyboardInterrupt happens in timely manner.
        """
        if platform.type != 'posix':
            raise SkipTest("I don't have the energy to fight Windows semantics.")
        program = "import os, threading, signal, time, sys\nimport crochet\ncrochet.setup()\nfrom twisted.internet.defer import Deferred\n\nif sys.platform.startswith('win'):\n    signal.signal(signal.SIGBREAK, signal.default_int_handler)\n    sig_int=signal.CTRL_BREAK_EVENT\n    sig_kill=signal.SIGTERM\nelse:\n    sig_int=signal.SIGINT\n    sig_kill=signal.SIGKILL\n\n\ndef interrupt():\n    time.sleep(0.1) # Make sure we've hit wait()\n    os.kill(os.getpid(), sig_int)\n    time.sleep(1)\n    # Still running, test shall fail...\n    os.kill(os.getpid(), sig_kill)\n\nt = threading.Thread(target=interrupt, daemon=True)\nt.start()\n\nd = Deferred()\ne = crochet.EventualResult(d, None)\n\ntry:\n    e.wait(10000)\nexcept KeyboardInterrupt:\n    sys.exit(23)\n"
        kw = {'cwd': crochet_directory}
        if platform.type.startswith('win'):
            kw['creationflags'] = subprocess.CREATE_NEW_PROCESS_GROUP
        process = subprocess.Popen([sys.executable, '-c', program], **kw)
        self.assertEqual(process.wait(), 23)

    def test_connect_deferred(self):
        """
        If an EventualResult is created with None,
        EventualResult._connect_deferred can be called later to register a
        Deferred as the one it is wrapping.
        """
        er = EventualResult(None, None)
        self.assertRaises(TimeoutError, er.wait, 0)
        d = Deferred()
        er._connect_deferred(d)
        self.assertRaises(TimeoutError, er.wait, 0)
        d.callback(123)
        self.assertEqual(er.wait(0.1), 123)

    def test_reactor_stop_unblocks_EventualResult(self):
        """
        Any EventualResult.wait() calls still waiting when the reactor has
        stopped will get a ReactorStopped exception.
        """
        program = 'import os, threading, signal, time, sys\n\nfrom twisted.internet.defer import Deferred\nfrom twisted.internet import reactor\n\nimport crochet\ncrochet.setup()\n\n@crochet.run_in_reactor\ndef run():\n    reactor.callLater(0.1, reactor.stop)\n    return Deferred()\n\ner = run()\ntry:\n    er.wait(timeout=10)\nexcept crochet.ReactorStopped:\n    sys.exit(23)\n'
        process = subprocess.Popen([sys.executable, '-c', program], cwd=crochet_directory)
        self.assertEqual(process.wait(), 23)

    def test_reactor_stop_unblocks_EventualResult_in_threadpool(self):
        """
        Any EventualResult.wait() calls still waiting when the reactor has
        stopped will get a ReactorStopped exception, even if it is running in
        Twisted's thread pool.
        """
        program = 'import os, threading, signal, time, sys\n\nfrom twisted.internet.defer import Deferred\nfrom twisted.internet import reactor\n\nimport crochet\ncrochet.setup()\n\n@crochet.run_in_reactor\ndef run():\n    reactor.callLater(0.1, reactor.stop)\n    return Deferred()\n\nresult = [13]\ndef inthread():\n    er = run()\n    try:\n        er.wait(timeout=10)\n    except crochet.ReactorStopped:\n        result[0] = 23\nreactor.callInThread(inthread)\ntime.sleep(1)\nsys.exit(result[0])\n'
        process = subprocess.Popen([sys.executable, '-c', program], cwd=crochet_directory)
        self.assertEqual(process.wait(), 23)

    def test_immediate_cancel(self):
        """
        Immediately cancelling the result of @run_in_reactor function will
        still cancel the Deferred.
        """
        program = 'import os, threading, signal, time, sys\n\nfrom twisted.internet.defer import Deferred, CancelledError\n\nimport crochet\ncrochet.setup()\n\n@crochet.run_in_reactor\ndef run():\n    return Deferred()\n\ner = run()\ner.cancel()\ntry:\n    er.wait(1)\nexcept CancelledError:\n    sys.exit(23)\nelse:\n    sys.exit(3)\n'
        process = subprocess.Popen([sys.executable, '-c', program], cwd=crochet_directory)
        self.assertEqual(process.wait(), 23)

    def test_noWaitingDuringImport(self):
        """
        EventualResult.wait() raises an exception if called while a module is
        being imported.

        This prevents the imports from taking a long time, preventing other
        imports from running in other threads. It also prevents deadlocks,
        which can happen if the code being waited on also tries to import
        something.
        """
        if sys.version_info[0] > 2:
            from unittest import SkipTest
            raise SkipTest('This test is too fragile (and insufficient) on Python 3 - see https://github.com/itamarst/crochet/issues/43')
        directory = tempfile.mktemp()
        os.mkdir(directory)
        sys.path.append(directory)
        self.addCleanup(sys.path.remove, directory)
        with open(os.path.join(directory, 'shouldbeunimportable.py'), 'w') as f:
            f.write('from crochet import EventualResult\nfrom twisted.internet.defer import Deferred\n\nEventualResult(Deferred(), None).wait(1.0)\n')
        self.assertRaises(RuntimeError, __import__, 'shouldbeunimportable')