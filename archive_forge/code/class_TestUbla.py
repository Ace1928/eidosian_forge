from __future__ import absolute_import
import re
from gslib.cs_api_map import ApiSelector
import gslib.tests.testcase as testcase
from gslib.tests.util import ObjectToURI as suri
from gslib.tests.util import unittest
from gslib.utils.retry_util import Retry
class TestUbla(testcase.GsUtilIntegrationTestCase):
    """Integration tests for ubla command."""
    _set_ubla_cmd = ['ubla', 'set']
    _get_ubla_cmd = ['ubla', 'get']

    def _AssertEnabled(self, bucket_uri, value):
        stdout = self.RunGsUtil(self._get_ubla_cmd + [suri(bucket_uri)], return_stdout=True)
        uniform_bucket_level_access_re = re.compile('^\\s*Enabled:\\s+(?P<enabled_val>.+)$', re.MULTILINE)
        uniform_bucket_level_access_match = re.search(uniform_bucket_level_access_re, stdout)
        uniform_bucket_level_access_val = uniform_bucket_level_access_match.group('enabled_val')
        self.assertEqual(str(value), uniform_bucket_level_access_val)

    def test_off_on_default_buckets(self):
        if self.test_api == ApiSelector.XML:
            return unittest.skip('XML API has no concept of Uniform bucket-level access')
        bucket_uri = self.CreateBucket()
        self._AssertEnabled(bucket_uri, False)

    def test_turning_off_on_enabled_buckets(self):
        if self.test_api == ApiSelector.XML:
            return unittest.skip('XML API has no concept of Uniform bucket-level access')
        bucket_uri = self.CreateBucket(bucket_policy_only=True, prefer_json_api=True)
        self._AssertEnabled(bucket_uri, True)
        self.RunGsUtil(self._set_ubla_cmd + ['off', suri(bucket_uri)])
        self._AssertEnabled(bucket_uri, False)

    def test_turning_on(self):
        if self.test_api == ApiSelector.XML:
            return unittest.skip('XML API has no concept of Uniform bucket-level access')
        bucket_uri = self.CreateBucket()
        self.RunGsUtil(self._set_ubla_cmd + ['on', suri(bucket_uri)])
        self._AssertEnabled(bucket_uri, True)

    def test_turning_on_and_off(self):
        if self.test_api == ApiSelector.XML:
            return unittest.skip('XML API has no concept of Uniform bucket-level access')
        bucket_uri = self.CreateBucket()
        self.RunGsUtil(self._set_ubla_cmd + ['on', suri(bucket_uri)])
        self._AssertEnabled(bucket_uri, True)
        self.RunGsUtil(self._set_ubla_cmd + ['off', suri(bucket_uri)])
        self._AssertEnabled(bucket_uri, False)

    def testTooFewArgumentsFails(self):
        """Ensures ubla commands fail with too few arguments."""
        stderr = self.RunGsUtil(self._set_ubla_cmd, return_stderr=True, expected_status=1)
        self.assertIn('command requires at least', stderr)
        stderr = self.RunGsUtil(self._get_ubla_cmd, return_stderr=True, expected_status=1)
        self.assertIn('command requires at least', stderr)
        stderr = self.RunGsUtil(['ubla'], return_stderr=True, expected_status=1)
        self.assertIn('command requires at least', stderr)