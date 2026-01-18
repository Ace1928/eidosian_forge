from __future__ import absolute_import
import os
import textwrap
from gslib.commands.rpo import RpoCommand
from gslib.exception import CommandException
from gslib.gcs_json_api import GcsJsonApi
from gslib.storage_url import StorageUrlFromString
import gslib.tests.testcase as testcase
from gslib.tests.testcase.integration_testcase import SkipForGS
from gslib.tests.testcase.integration_testcase import SkipForJSON
from gslib.tests.testcase.integration_testcase import SkipForXML
from gslib.tests.util import ObjectToURI as suri
from gslib.tests.util import SetBotoConfigForTest
from gslib.tests.util import SetEnvironmentForTest
from gslib.utils import shim_util
from six import add_move, MovedModule
from six.moves import mock
class TestRpoE2E(testcase.GsUtilIntegrationTestCase):
    """Integration tests for rpo command."""

    def _verify_get_returns_default_or_none(self, bucket_uri):
        """Checks if the rpo get command returns default."""
        try:
            self.VerifyCommandGet(bucket_uri, 'rpo', 'DEFAULT')
        except AssertionError:
            self.VerifyCommandGet(bucket_uri, 'rpo', 'None')

    @SkipForXML('RPO only runs on GCS JSON API.')
    def test_get_returns_default_for_dual_region_bucket(self):
        bucket_uri = self.CreateBucket(location='nam4')
        self._verify_get_returns_default_or_none(bucket_uri)

    @SkipForXML('RPO only runs on GCS JSON API.')
    def test_get_returns_none_for_regional_bucket(self):
        bucket_uri = self.CreateBucket(location='us-central1')
        self.VerifyCommandGet(bucket_uri, 'rpo', 'None')

    @SkipForXML('RPO only runs on GCS JSON API.')
    def test_set_and_get_async_turbo(self):
        bucket_uri = self.CreateBucket(location='nam4')
        self._verify_get_returns_default_or_none(bucket_uri)
        self.RunGsUtil(['rpo', 'set', 'ASYNC_TURBO', suri(bucket_uri)])
        self.VerifyCommandGet(bucket_uri, 'rpo', 'ASYNC_TURBO')

    @SkipForXML('RPO only runs on GCS JSON API.')
    def test_set_default(self):
        bucket_uri = self.CreateBucket(location='nam4')
        self.RunGsUtil(['rpo', 'set', 'ASYNC_TURBO', suri(bucket_uri)])
        self.VerifyCommandGet(bucket_uri, 'rpo', 'ASYNC_TURBO')
        self.RunGsUtil(['rpo', 'set', 'DEFAULT', suri(bucket_uri)])
        self._verify_get_returns_default_or_none(bucket_uri)

    @SkipForXML('RPO only runs on GCS JSON API.')
    def test_set_async_turbo_fails_for_regional_buckets(self):
        bucket_uri = self.CreateBucket(location='us-central1')
        stderr = self.RunGsUtil(['rpo', 'set', 'ASYNC_TURBO', suri(bucket_uri)], expected_status=1, return_stderr=True)
        self.assertIn('ASYNC_TURBO cannot be enabled on REGION bucket', stderr)

    @SkipForJSON('Testing XML only behavior.')
    def test_xml_fails_for_set(self):
        boto_config_hmac_auth_only = [('Credentials', 'gs_oauth2_refresh_token', None), ('Credentials', 'gs_service_client_id', None), ('Credentials', 'gs_service_key_file', None), ('Credentials', 'gs_service_key_file_password', None), ('Credentials', 'gs_access_key_id', 'dummykey'), ('Credentials', 'gs_secret_access_key', 'dummysecret')]
        with SetBotoConfigForTest(boto_config_hmac_auth_only):
            bucket_uri = 'gs://any-bucket-name'
            stderr = self.RunGsUtil(['rpo', 'set', 'default', bucket_uri], return_stderr=True, expected_status=1)
            self.assertIn('command can only be with the Cloud Storage JSON API', stderr)

    @SkipForJSON('Testing XML only behavior.')
    def test_xml_fails_for_get(self):
        boto_config_hmac_auth_only = [('Credentials', 'gs_oauth2_refresh_token', None), ('Credentials', 'gs_service_client_id', None), ('Credentials', 'gs_service_key_file', None), ('Credentials', 'gs_service_key_file_password', None), ('Credentials', 'gs_access_key_id', 'dummykey'), ('Credentials', 'gs_secret_access_key', 'dummysecret')]
        with SetBotoConfigForTest(boto_config_hmac_auth_only):
            bucket_uri = 'gs://any-bucket-name'
            stderr = self.RunGsUtil(['rpo', 'get', bucket_uri], return_stderr=True, expected_status=1)
            self.assertIn('command can only be with the Cloud Storage JSON API', stderr)

    @SkipForGS('Testing S3 only behavior.')
    def test_s3_fails_for_set(self):
        bucket_uri = self.CreateBucket()
        stderr = self.RunGsUtil(['rpo', 'set', 'DEFAULT', suri(bucket_uri)], return_stderr=True, expected_status=1)
        if self._use_gcloud_storage:
            self.assertIn('Features disallowed for S3: Setting Recovery Point Objective', stderr)
        else:
            self.assertIn('command can only be used for GCS buckets', stderr)

    @SkipForGS('Testing S3 only behavior.')
    def test_s3_fails_for_get(self):
        bucket_uri = self.CreateBucket()
        expected_status = 0 if self._use_gcloud_storage else 1
        stdout, stderr = self.RunGsUtil(['rpo', 'get', suri(bucket_uri)], return_stderr=True, return_stdout=True, expected_status=expected_status)
        if self._use_gcloud_storage:
            self.assertIn('gs://None: None', stdout)
        else:
            self.assertIn('command can only be used for GCS buckets', stderr)