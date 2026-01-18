import datetime
import os
import stat
from contextlib import contextmanager
from io import BytesIO
from itertools import permutations
from dulwich.tests import TestCase
from ..errors import ObjectFormatException
from ..objects import (
from .utils import ext_functest_builder, functest_builder, make_commit, make_object
class CheckTests(TestCase):

    def test_check_hexsha(self):
        check_hexsha(a_sha, 'failed to check good sha')
        self.assertRaises(ObjectFormatException, check_hexsha, b'1' * 39, 'sha too short')
        self.assertRaises(ObjectFormatException, check_hexsha, b'1' * 41, 'sha too long')
        self.assertRaises(ObjectFormatException, check_hexsha, b'x' * 40, 'invalid characters')

    def test_check_identity(self):
        check_identity(b'Dave Borowitz <dborowitz@google.com>', 'failed to check good identity')
        check_identity(b' <dborowitz@google.com>', 'failed to check good identity')
        self.assertRaises(ObjectFormatException, check_identity, b'<dborowitz@google.com>', 'no space before email')
        self.assertRaises(ObjectFormatException, check_identity, b'Dave Borowitz', 'no email')
        self.assertRaises(ObjectFormatException, check_identity, b'Dave Borowitz <dborowitz', 'incomplete email')
        self.assertRaises(ObjectFormatException, check_identity, b'dborowitz@google.com>', 'incomplete email')
        self.assertRaises(ObjectFormatException, check_identity, b'Dave Borowitz <<dborowitz@google.com>', 'typo')
        self.assertRaises(ObjectFormatException, check_identity, b'Dave Borowitz <dborowitz@google.com>>', 'typo')
        self.assertRaises(ObjectFormatException, check_identity, b'Dave Borowitz <dborowitz@google.com>xxx', 'trailing characters')
        self.assertRaises(ObjectFormatException, check_identity, b'Dave Borowitz <dborowitz@google.com>xxx', 'trailing characters')
        self.assertRaises(ObjectFormatException, check_identity, b'Dave<Borowitz <dborowitz@google.com>', 'reserved byte in name')
        self.assertRaises(ObjectFormatException, check_identity, b'Dave>Borowitz <dborowitz@google.com>', 'reserved byte in name')
        self.assertRaises(ObjectFormatException, check_identity, b'Dave\x00Borowitz <dborowitz@google.com>', 'null byte')
        self.assertRaises(ObjectFormatException, check_identity, b'Dave\nBorowitz <dborowitz@google.com>', 'newline byte')