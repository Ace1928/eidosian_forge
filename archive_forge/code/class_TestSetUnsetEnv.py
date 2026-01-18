import errno
import os
import select
import socket
import sys
import tempfile
import time
from io import BytesIO
from .. import errors, osutils, tests, trace, win32utils
from . import features, file_utils, test__walkdirs_win32
from .scenarios import load_tests_apply_scenarios
class TestSetUnsetEnv(tests.TestCase):
    """Test updating the environment"""

    def setUp(self):
        super().setUp()
        self.assertEqual(None, os.environ.get('BRZ_TEST_ENV_VAR'), 'Environment was not cleaned up properly. Variable BRZ_TEST_ENV_VAR should not exist.')

        def cleanup():
            if 'BRZ_TEST_ENV_VAR' in os.environ:
                del os.environ['BRZ_TEST_ENV_VAR']
        self.addCleanup(cleanup)

    def test_set(self):
        """Test that we can set an env variable"""
        old = osutils.set_or_unset_env('BRZ_TEST_ENV_VAR', 'foo')
        self.assertEqual(None, old)
        self.assertEqual('foo', os.environ.get('BRZ_TEST_ENV_VAR'))

    def test_double_set(self):
        """Test that we get the old value out"""
        osutils.set_or_unset_env('BRZ_TEST_ENV_VAR', 'foo')
        old = osutils.set_or_unset_env('BRZ_TEST_ENV_VAR', 'bar')
        self.assertEqual('foo', old)
        self.assertEqual('bar', os.environ.get('BRZ_TEST_ENV_VAR'))

    def test_unicode(self):
        """Environment can only contain plain strings

        So Unicode strings must be encoded.
        """
        uni_val, env_val = tests.probe_unicode_in_user_encoding()
        if uni_val is None:
            raise tests.TestSkipped('Cannot find a unicode character that works in encoding %s' % (osutils.get_user_encoding(),))
        osutils.set_or_unset_env('BRZ_TEST_ENV_VAR', uni_val)
        self.assertEqual(uni_val, os.environ.get('BRZ_TEST_ENV_VAR'))

    def test_unset(self):
        """Test that passing None will remove the env var"""
        osutils.set_or_unset_env('BRZ_TEST_ENV_VAR', 'foo')
        old = osutils.set_or_unset_env('BRZ_TEST_ENV_VAR', None)
        self.assertEqual('foo', old)
        self.assertEqual(None, os.environ.get('BRZ_TEST_ENV_VAR'))
        self.assertNotIn('BRZ_TEST_ENV_VAR', os.environ)