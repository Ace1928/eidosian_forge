import gc
import io
import os
import sys
import signal
import weakref
import unittest
from test import support
@unittest.skipUnless(hasattr(os, 'kill'), 'Test requires os.kill')
@unittest.skipIf(sys.platform == 'win32', 'Test cannot run on Windows')
class TestBreakSignalIgnored(TestBreak):
    int_handler = signal.SIG_IGN