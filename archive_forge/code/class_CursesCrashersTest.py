import fcntl
import os
import pty
import struct
import sys
import termios
import textwrap
import unittest
from bpython.test import TEST_CONFIG
from bpython.config import getpreferredencoding
@unittest.skipIf(reactor is None, 'twisted is not available')
class CursesCrashersTest(TrialTestCase, CrashersTest):
    backend = 'cli'