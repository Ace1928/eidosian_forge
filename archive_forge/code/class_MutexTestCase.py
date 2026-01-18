import threading
import uuid
from os_win import exceptions
from os_win.tests.functional import test_base
from os_win.utils import processutils
class MutexTestCase(test_base.OsWinBaseFunctionalTestCase):

    def setUp(self):
        super(MutexTestCase, self).setUp()
        mutex_name = str(uuid.uuid4())
        self._mutex = processutils.Mutex(name=mutex_name)
        self.addCleanup(self._mutex.close)

    def acquire_mutex_in_separate_thread(self, mutex):
        stop_event = threading.Event()

        def target():
            mutex.acquire()
            stop_event.wait()
            mutex.release()
        thread = threading.Thread(target=target)
        thread.daemon = True
        thread.start()
        return (thread, stop_event)

    def test_already_acquired_mutex(self):
        thread, stop_event = self.acquire_mutex_in_separate_thread(self._mutex)
        self.assertFalse(self._mutex.acquire(timeout_ms=0))
        stop_event.set()
        self.assertTrue(self._mutex.acquire(timeout_ms=2000))

    def test_release_unacquired_mutex(self):
        self.assertRaises(exceptions.Win32Exception, self._mutex.release)

    def test_multiple_acquire(self):
        self._mutex.acquire(timeout_ms=0)
        self._mutex.acquire(timeout_ms=0)
        self._mutex.release()
        self._mutex.release()