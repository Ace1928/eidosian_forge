import threading
import fasteners
from fasteners import test
class HelpersTest(test.TestCase):

    def test_try_lock(self):
        lock = threading.Lock()
        with fasteners.try_lock(lock) as locked:
            self.assertTrue(locked)
            with fasteners.try_lock(lock) as locked:
                self.assertFalse(locked)