from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import os
from random import randint
from unittest import mock
from gslib.cloud_api import AccessDeniedException
from gslib.project_id import PopulateProjectId
import gslib.tests.testcase as testcase
from gslib.tests.testcase.integration_testcase import SkipForJSON
from gslib.tests.testcase.integration_testcase import SkipForS3
from gslib.tests.testcase.integration_testcase import SkipForXML
from gslib.tests.util import ObjectToURI as suri
from gslib.tests.util import SetBotoConfigForTest
from gslib.tests.util import SetEnvironmentForTest
from gslib.utils.retry_util import Retry
from gslib.utils import shim_util
@SkipForS3('gsutil does not support KMS operations for S3 buckets.')
@SkipForJSON('These tests only check for failures when the XML API is forced.')
class TestKmsSubcommandsFailWhenXmlForced(testcase.GsUtilIntegrationTestCase):
    """Tests that kms subcommands fail early when forced to use the XML API."""
    boto_config_hmac_auth_only = [('Credentials', 'gs_oauth2_refresh_token', None), ('Credentials', 'gs_service_client_id', None), ('Credentials', 'gs_service_key_file', None), ('Credentials', 'gs_service_key_file_password', None), ('Credentials', 'gs_access_key_id', 'dummykey'), ('Credentials', 'gs_secret_access_key', 'dummysecret')]
    dummy_keyname = 'projects/my-project/locations/us-central1/keyRings/my-keyring/cryptoKeys/my-key'

    def DoTestSubcommandFailsWhenXmlForcedFromHmacInBotoConfig(self, subcommand):
        with SetBotoConfigForTest(self.boto_config_hmac_auth_only):
            stderr = self.RunGsUtil(subcommand, expected_status=1, return_stderr=True)
            self.assertIn('The "kms" command can only be used with', stderr)

    def testEncryptionFailsWhenXmlForcedFromHmacInBotoConfig(self):
        self.DoTestSubcommandFailsWhenXmlForcedFromHmacInBotoConfig(['kms', 'encryption', 'gs://dummybucket'])

    def testEncryptionDashKFailsWhenXmlForcedFromHmacInBotoConfig(self):
        self.DoTestSubcommandFailsWhenXmlForcedFromHmacInBotoConfig(['kms', 'encryption', '-k', self.dummy_keyname, 'gs://dummybucket'])

    def testEncryptionDashDFailsWhenXmlForcedFromHmacInBotoConfig(self):
        self.DoTestSubcommandFailsWhenXmlForcedFromHmacInBotoConfig(['kms', 'encryption', '-d', 'gs://dummybucket'])

    def testServiceaccountFailsWhenXmlForcedFromHmacInBotoConfig(self):
        self.DoTestSubcommandFailsWhenXmlForcedFromHmacInBotoConfig(['kms', 'serviceaccount', 'gs://dummybucket'])

    def testAuthorizeFailsWhenXmlForcedFromHmacInBotoConfig(self):
        self.DoTestSubcommandFailsWhenXmlForcedFromHmacInBotoConfig(['kms', 'authorize', '-k', self.dummy_keyname, 'gs://dummybucket'])