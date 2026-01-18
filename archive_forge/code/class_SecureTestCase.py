import reportlab
import sys, os, fnmatch, re, functools
from configparser import ConfigParser
import unittest
from reportlab.lib.utils import isCompactDistro, __rl_loader__, rl_isdir, asUnicode
class SecureTestCase(unittest.TestCase):
    """Secure testing base class with additional pre- and postconditions.

    We try to ensure that each test leaves the environment it has
    found unchanged after the test is performed, successful or not.

    Currently we restore sys.path and the working directory, but more
    of this could be added easily, like removing temporary files or
    similar things.

    Use this as a base class replacing unittest.TestCase and call
    these methods in subclassed versions before doing your own
    business!
    """

    def setUp(self):
        """Remember sys.path and current working directory."""
        self._initialPath = sys.path[:]
        self._initialWorkDir = os.getcwd()

    def tearDown(self):
        """Restore previous sys.path and working directory."""
        sys.path = self._initialPath
        os.chdir(self._initialWorkDir)