import inspect
import re
import socket
import sys
from .. import controldir, errors, osutils, tests, urlutils
class TestErrorFormatting(tests.TestCase):

    def test_always_str(self):
        e = PassThroughError('µ', 'bar')
        self.assertIsInstance(e.__str__(), str)
        s = str(e)
        self.assertEqual('Pass through µ and bar', s)

    def test_missing_format_string(self):
        e = ErrorWithNoFormat(param='randomvalue')
        self.assertStartsWith(str(e), 'Unprintable exception ErrorWithNoFormat')

    def test_mismatched_format_args(self):
        e = ErrorWithBadFormat(not_thing='x')
        self.assertStartsWith(str(e), 'Unprintable exception ErrorWithBadFormat')

    def test_cannot_bind_address(self):
        e = errors.CannotBindAddress('example.com', 22, socket.error(13, 'Permission denied'))
        self.assertContainsRe(str(e), 'Cannot bind address "example\\.com:22":.*Permission denied')