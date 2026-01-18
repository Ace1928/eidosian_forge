from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
from datetime import datetime
from datetime import timedelta
import os
import pkgutil
import boto
import gslib.commands.signurl
from gslib.commands.signurl import HAVE_OPENSSL
from gslib.exception import CommandException
from gslib.gcs_json_api import GcsJsonApi
from gslib.iamcredentials_api import IamcredentailsApi
from gslib.impersonation_credentials import ImpersonationCredentials
import gslib.tests.testcase as testcase
from gslib.tests.testcase.integration_testcase import (SkipForS3, SkipForXML)
from gslib.tests.util import ObjectToURI as suri
from gslib.tests.util import SetBotoConfigForTest
from gslib.tests.util import SetEnvironmentForTest
from gslib.tests.util import unittest
import gslib.tests.signurl_signatures as sigs
from oauth2client import client
from oauth2client.service_account import ServiceAccountCredentials
from six import add_move, MovedModule
from six.moves import mock
@unittest.skipUnless(HAVE_OPENSSL, 'signurl requires pyopenssl.')
@SkipForS3('Signed URLs are only supported for gs:// URLs.')
class TestSignUrl(testcase.GsUtilIntegrationTestCase):
    """Integration tests for signurl command."""

    def _GetJSONKsFile(self):
        if not hasattr(self, 'json_ks_file'):
            contents = pkgutil.get_data('gslib', 'tests/test_data/test.json')
            self.json_ks_file = self.CreateTempFile(contents=contents)
        return self.json_ks_file

    def _GetKsFile(self):
        if not hasattr(self, 'ks_file'):
            contents = pkgutil.get_data('gslib', 'tests/test_data/test.p12')
            self.ks_file = self.CreateTempFile(contents=contents)
        return self.ks_file

    def testSignUrlInvalidDuration(self):
        """Tests signurl fails with out of bounds value for valid duration."""
        if self._use_gcloud_storage:
            expected_status = 2
        else:
            expected_status = 1
        stderr = self.RunGsUtil(['signurl', '-d', '123d', 'ks_file', 'gs://uri'], return_stderr=True, expected_status=expected_status)
        if self._use_gcloud_storage:
            self.assertIn('value must be less than or equal to 7d', stderr)
        else:
            self.assertIn('CommandException: Max valid duration allowed is 7 days', stderr)

    def testSignUrlInvalidDurationWithUseServiceAccount(self):
        """Tests signurl with -u flag fails duration > 12 hours."""
        stderr = self.RunGsUtil(['signurl', '-d', '13h', '-u', 'gs://uri'], return_stderr=True, expected_status=1)
        self.assertIn('CommandException: Max valid duration allowed is 12:00:00', stderr)

    def testSignUrlOutputP12(self):
        """Tests signurl output of a sample object with pkcs12 keystore."""
        bucket_uri = self.CreateBucket()
        object_uri = self.CreateObject(bucket_uri=bucket_uri, contents=b'z')
        cmd = ['signurl', '-p', 'notasecret', '-m', 'PUT', self._GetKsFile(), suri(object_uri)]
        stdout = self.RunGsUtil(cmd, return_stdout=True)
        self.assertIn('x-goog-credential=test.apps.googleusercontent.com', stdout)
        self.assertIn('x-goog-expires=3600', stdout)
        self.assertIn('%2Fus-central1%2F', stdout)
        self.assertIn('\tPUT\t', stdout)

    def testSignUrlOutputJSON(self):
        """Tests signurl output of a sample object with JSON keystore."""
        bucket_uri = self.CreateBucket()
        object_uri = self.CreateObject(bucket_uri=bucket_uri, contents=b'z')
        cmd = ['signurl', '-m', 'PUT', self._GetJSONKsFile(), suri(object_uri)]
        stdout = self.RunGsUtil(cmd, return_stdout=True)
        self.assertIn('x-goog-credential=' + TEST_EMAIL, stdout)
        self.assertIn('x-goog-expires=3600', stdout)
        self.assertIn('%2Fus-central1%2F', stdout)
        self.assertIn('\tPUT\t', stdout)

    def testSignUrlWithJSONKeyFileAndObjectGeneration(self):
        """Tests signurl output of a sample object version with JSON keystore."""
        bucket_uri = self.CreateBucket(versioning_enabled=True)
        object_uri = self.CreateObject(bucket_uri=bucket_uri, contents=b'z')
        cmd = ['signurl', self._GetJSONKsFile(), object_uri.version_specific_uri]
        stdout = self.RunGsUtil(cmd, return_stdout=True)
        self.assertIn('x-goog-credential=' + TEST_EMAIL, stdout)
        self.assertIn('generation=' + object_uri.generation, stdout)

    def testSignUrlWithURLEncodeRequiredChars(self):
        objs = ['gs://example.org/test 1', 'gs://example.org/test/test 2', 'gs://example.org/Аудиоарi хив']
        expected_partial_urls = ['https://storage.googleapis.com/example.org/test%201?x-goog-signature=', 'https://storage.googleapis.com/example.org/test/test%202?x-goog-signature=', 'https://storage.googleapis.com/example.org/%D0%90%D1%83%D0%B4%D0%B8%D0%BE%D0%B0%D1%80i%20%D1%85%D0%B8%D0%B2?x-goog-signature=']
        self.assertEqual(len(objs), len(expected_partial_urls))
        cmd_args = ['signurl', '-m', 'PUT', '-p', 'notasecret', '-r', 'us', self._GetKsFile()]
        cmd_args.extend(objs)
        stdout = self.RunGsUtil(cmd_args, return_stdout=True)
        lines = stdout.split('\n')
        self.assertEqual(len(lines), len(objs) + 2)
        lines = lines[1:]
        for obj, line, partial_url in zip(objs, lines, expected_partial_urls):
            self.assertIn(obj, line)
            self.assertIn(partial_url, line)
            self.assertIn('x-goog-credential=test.apps.googleusercontent.com', line)
        self.assertIn('%2Fus%2F', stdout)

    def testSignUrlWithWildcard(self):
        objs = ['test1', 'test2', 'test3']
        obj_urls = []
        bucket = self.CreateBucket()
        for obj_name in objs:
            obj_urls.append(self.CreateObject(bucket_uri=bucket, object_name=obj_name, contents=b''))
        stdout = self.RunGsUtil(['signurl', '-p', 'notasecret', self._GetKsFile(), suri(bucket) + '/*'], return_stdout=True)
        self.assertEqual(len(stdout.split('\n')), 5)
        for obj_url in obj_urls:
            self.assertIn(suri(obj_url), stdout)

    @unittest.skipUnless(SERVICE_ACCOUNT, 'Test requires test_impersonate_service_account.')
    @SkipForS3('Tests only uses gs credentials.')
    @SkipForXML('Tests only run on JSON API.')
    def testSignUrlWithServiceAccount(self):
        with SetBotoConfigForTest([('Credentials', 'gs_impersonate_service_account', SERVICE_ACCOUNT)]):
            stdout, stderr = self.RunGsUtil(['signurl', '-r', 'us-east1', '-u', 'gs://pub'], return_stdout=True, return_stderr=True)
        self.assertIn('https://storage.googleapis.com/pub', stdout)
        self.assertIn('All API calls will be executed as [%s]' % SERVICE_ACCOUNT, stderr)

    def testSignUrlOfNonObjectUrl(self):
        """Tests the signurl output of a non-existent file."""
        self.RunGsUtil(['signurl', self._GetKsFile(), 'gs://'], expected_status=1, stdin='notasecret')
        self.RunGsUtil(['signurl', 'file://tmp/abc', 'gs://bucket'], expected_status=1)