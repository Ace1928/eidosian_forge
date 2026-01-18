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
@SkipForXML('gsutil does not support KMS operations for S3 buckets.')
class TestKmsSuccessCases(testcase.GsUtilIntegrationTestCase):
    """Integration tests for the kms command."""

    def setUp(self):
        super(TestKmsSuccessCases, self).setUp()
        self.keyring_fqn = self.kms_api.CreateKeyRing(PopulateProjectId(None), testcase.KmsTestingResources.KEYRING_NAME, location=testcase.KmsTestingResources.KEYRING_LOCATION)

    @Retry(AssertionError, tries=3, timeout_secs=1)
    def DoTestAuthorize(self, specified_project=None):
        key_name = testcase.KmsTestingResources.MUTABLE_KEY_NAME_TEMPLATE % (randint(0, 9), randint(0, 9), randint(0, 9))
        key_fqn = self.kms_api.CreateCryptoKey(self.keyring_fqn, key_name)
        key_policy = self.kms_api.GetKeyIamPolicy(key_fqn)
        while key_policy.bindings:
            key_policy.bindings.pop()
        self.kms_api.SetKeyIamPolicy(key_fqn, key_policy)
        authorize_cmd = ['kms', 'authorize', '-k', key_fqn]
        if specified_project:
            authorize_cmd.extend(['-p', specified_project])
        stdout1 = self.RunGsUtil(authorize_cmd, return_stdout=True)
        stdout2 = self.RunGsUtil(authorize_cmd, return_stdout=True)
        self.assertIn('Authorized project %s to encrypt and decrypt with key:\n%s' % (PopulateProjectId(None), key_fqn), stdout1)
        self.assertIn('Project %s was already authorized to encrypt and decrypt with key:\n%s.' % (PopulateProjectId(None), key_fqn), stdout2)

    def DoTestServiceaccount(self, specified_project=None):
        serviceaccount_cmd = ['kms', 'serviceaccount']
        if specified_project:
            serviceaccount_cmd.extend(['-p', specified_project])
        stdout = self.RunGsUtil(serviceaccount_cmd, return_stdout=True)
        self.assertRegex(stdout, '[^@]+@gs-project-accounts\\.iam\\.gserviceaccount\\.com')

    def testKmsAuthorizeWithoutProjectOption(self):
        self.DoTestAuthorize()

    def testKmsAuthorizeWithProjectOption(self):
        self.DoTestAuthorize(specified_project=PopulateProjectId(None))

    def testKmsServiceaccountWithoutProjectOption(self):
        self.DoTestServiceaccount()

    def testKmsServiceaccountWithProjectOption(self):
        self.DoTestServiceaccount(specified_project=PopulateProjectId(None))

    def testKmsEncryptionFlow(self):
        bucket_uri = self.CreateBucket()
        key_fqn = self.kms_api.CreateCryptoKey(self.keyring_fqn, testcase.KmsTestingResources.CONSTANT_KEY_NAME)
        encryption_get_cmd = ['kms', 'encryption', suri(bucket_uri)]
        stdout = self.RunGsUtil(encryption_get_cmd, return_stdout=True)
        self.assertIn('Bucket %s has no default encryption key' % suri(bucket_uri), stdout)
        stdout = self.RunGsUtil(['kms', 'encryption', '-k', key_fqn, suri(bucket_uri)], return_stdout=True)
        self.assertIn('Setting default KMS key for bucket %s...' % suri(bucket_uri), stdout)
        stdout = self.RunGsUtil(encryption_get_cmd, return_stdout=True)
        self.assertIn('Default encryption key for %s:\n%s' % (suri(bucket_uri), key_fqn), stdout)
        stdout = self.RunGsUtil(['kms', 'encryption', '-d', suri(bucket_uri)], return_stdout=True)
        self.assertIn('Clearing default encryption key for %s...' % suri(bucket_uri), stdout)
        stdout = self.RunGsUtil(encryption_get_cmd, return_stdout=True)
        self.assertIn('Bucket %s has no default encryption key' % suri(bucket_uri), stdout)