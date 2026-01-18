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