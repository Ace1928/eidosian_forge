import os
import sys
import tempfile
from io import BytesIO
from typing import ClassVar, Dict
from dulwich import errors
from dulwich.tests import SkipTest, TestCase
from ..file import GitFile
from ..objects import ZERO_SHA
from ..refs import (
from ..repo import Repo
from .utils import open_repo, tear_down_repo
class CheckRefFormatTests(TestCase):
    """Tests for the check_ref_format function.

    These are the same tests as in the git test suite.
    """

    def test_valid(self):
        self.assertTrue(check_ref_format(b'heads/foo'))
        self.assertTrue(check_ref_format(b'foo/bar/baz'))
        self.assertTrue(check_ref_format(b'refs///heads/foo'))
        self.assertTrue(check_ref_format(b'foo./bar'))
        self.assertTrue(check_ref_format(b'heads/foo@bar'))
        self.assertTrue(check_ref_format(b'heads/fix.lock.error'))

    def test_invalid(self):
        self.assertFalse(check_ref_format(b'foo'))
        self.assertFalse(check_ref_format(b'heads/foo/'))
        self.assertFalse(check_ref_format(b'./foo'))
        self.assertFalse(check_ref_format(b'.refs/foo'))
        self.assertFalse(check_ref_format(b'heads/foo..bar'))
        self.assertFalse(check_ref_format(b'heads/foo?bar'))
        self.assertFalse(check_ref_format(b'heads/foo.lock'))
        self.assertFalse(check_ref_format(b'heads/v@{ation'))
        self.assertFalse(check_ref_format(b'heads/foo\x08ar'))