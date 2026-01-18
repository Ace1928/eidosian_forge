from __future__ import annotations
import re
import sys
import traceback
from twisted.python.compat import nativeString
from twisted.trial import unittest
from twisted.web import error
from twisted.web.template import Tag
class UnsupportedMethodTests(unittest.SynchronousTestCase):
    """
    Tests for L{UnsupportedMethod}.
    """

    def test_str(self) -> None:
        """
        The C{__str__} for L{UnsupportedMethod} makes it clear that what it
        shows is a list of the supported methods, not the method that was
        unsupported.
        """
        e = error.UnsupportedMethod([b'HEAD', b'PATCH'])
        self.assertEqual(str(e), "Expected one of [b'HEAD', b'PATCH']")