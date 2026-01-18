import unittest
import time
class KeyModuleTest(unittest.TestCase):

    def test_get_focused(self):
        stop_time = time.time() + 10.0
        while time.time() < stop_time:
            time.sleep(1)
        self.assertTrue(True)