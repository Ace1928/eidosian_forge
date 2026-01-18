import codecs
import io
import sys
import traceback
from unittest import skipIf
from twisted.python.compat import (
from twisted.python.filepath import FilePath
from twisted.python.runtime import platform
from twisted.trial.unittest import SynchronousTestCase, TestCase
class BytesEnvironTests(TestCase):
    """
    Tests for L{BytesEnviron}.
    """

    @skipIf(platform.isWindows(), 'Environment vars are always str on Windows.')
    def test_alwaysBytes(self):
        """
        The output of L{BytesEnviron} should always be a L{dict} with L{bytes}
        values and L{bytes} keys.
        """
        result = bytesEnviron()
        types = set()
        for key, val in result.items():
            types.add(type(key))
            types.add(type(val))
        self.assertEqual(list(types), [bytes])