import gzip
import os
import tempfile
from .... import tests
from ..exporter import (_get_output_stream, check_ref_format,
from . import FastimportFeature
class CheckRefnameRewriting(tests.TestCase):
    """Tests for sanitize_ref_name_for_git function"""

    def test_passthrough_valid(self):
        self.assertEqual(sanitize_ref_name_for_git(b'heads/foo'), b'heads/foo')
        self.assertEqual(sanitize_ref_name_for_git(b'foo/bar/baz'), b'foo/bar/baz')
        self.assertEqual(sanitize_ref_name_for_git(b'refs///heads/foo'), b'refs///heads/foo')
        self.assertEqual(sanitize_ref_name_for_git(b'foo./bar'), b'foo./bar')
        self.assertEqual(sanitize_ref_name_for_git(b'heads/foo@bar'), b'heads/foo@bar')
        self.assertEqual(sanitize_ref_name_for_git(b'heads/fix.lock.error'), b'heads/fix.lock.error')

    def test_rewrite_invalid(self):
        self.assertTrue(check_ref_format(sanitize_ref_name_for_git(b'foo./bar')))
        self.assertTrue(check_ref_format(sanitize_ref_name_for_git(b'heads/foo/')))
        self.assertTrue(check_ref_format(sanitize_ref_name_for_git(b'heads/foo.')))
        self.assertTrue(check_ref_format(sanitize_ref_name_for_git(b'./foo')))
        self.assertTrue(check_ref_format(sanitize_ref_name_for_git(b'.refs/foo')))
        self.assertTrue(check_ref_format(sanitize_ref_name_for_git(b'heads/foo..bar')))
        self.assertTrue(check_ref_format(sanitize_ref_name_for_git(b'heads/foo?bar')))
        self.assertTrue(check_ref_format(sanitize_ref_name_for_git(b'heads/foo.lock')))
        self.assertTrue(check_ref_format(sanitize_ref_name_for_git(b'heads/v@{ation')))
        self.assertTrue(check_ref_format(sanitize_ref_name_for_git(b'heads/foo\x08ar')))
        self.assertTrue(check_ref_format(sanitize_ref_name_for_git(b'heads/foo\\bar')))
        self.assertTrue(check_ref_format(sanitize_ref_name_for_git(b'heads/foo bar')))
        self.assertTrue(check_ref_format(sanitize_ref_name_for_git(b'heads/foo\x10bar')))
        self.assertTrue(check_ref_format(sanitize_ref_name_for_git(b'heads/foo\x7fbar')))