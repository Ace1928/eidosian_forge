from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import os
from random import randint
import boto
import gslib.tests.testcase as testcase
from gslib.project_id import PopulateProjectId
from gslib.tests.testcase.integration_testcase import SkipForS3
from gslib.tests.testcase.integration_testcase import SkipForXML
from gslib.tests.testcase.integration_testcase import SkipForJSON
from gslib.tests.util import ObjectToURI as suri
from gslib.utils.retention_util import SECONDS_IN_DAY
from gslib.utils.retention_util import SECONDS_IN_MONTH
from gslib.utils.retention_util import SECONDS_IN_YEAR
from gslib.tests.util import SetBotoConfigForTest
from gslib.tests.util import SetEnvironmentForTest
from gslib.utils.retry_util import Retry
from gslib.utils import shim_util
class TestMb(testcase.GsUtilIntegrationTestCase):
    """Integration tests for mb command."""

    def GetKey(self, mutable=False):
        keyring_fqn = self.kms_api.CreateKeyRing(PopulateProjectId(None), testcase.KmsTestingResources.KEYRING_NAME, location=testcase.KmsTestingResources.KEYRING_LOCATION)
        key_name = testcase.KmsTestingResources.CONSTANT_KEY_NAME_DO_NOT_AUTHORIZE
        if mutable:
            key_name = testcase.KmsTestingResources.MUTABLE_KEY_NAME_TEMPLATE % (randint(0, 9), randint(0, 9), randint(0, 9))
        key_fqn = self.kms_api.CreateCryptoKey(keyring_fqn, key_name)
        key_policy = self.kms_api.GetKeyIamPolicy(key_fqn)
        if key_policy.bindings:
            key_policy.bindings = []
            self.kms_api.SetKeyIamPolicy(key_fqn, key_policy)
        return key_fqn

    @SkipForS3('S3 returns success when bucket already exists.')
    def test_mb_bucket_exists(self):
        bucket_uri = self.CreateBucket()
        stderr = self.RunGsUtil(['mb', suri(bucket_uri)], expected_status=1, return_stderr=True)
        if self._use_gcloud_storage:
            self.assertIn('HTTPError 409: The requested bucket name is not available.', stderr)
        else:
            self.assertIn('already exists', stderr)

    def test_non_ascii_project_fails(self):
        stderr = self.RunGsUtil(['mb', '-p', 'ã', 'gs://fobarbaz'], expected_status=1, return_stderr=True)
        if self._use_gcloud_storage:
            self.assertIn('The project property must be set to a valid project ID', stderr)
        else:
            self.assertIn('Invalid non-ASCII', stderr)

    @SkipForS3(BUCKET_LOCK_SKIP_MSG)
    def test_create_with_retention_seconds(self):
        bucket_name = self.MakeTempName('bucket')
        bucket_uri = boto.storage_uri('gs://%s' % bucket_name.lower(), suppress_consec_slashes=False)
        self.RunGsUtil(['mb', '--retention', '60s', suri(bucket_uri)])
        self.VerifyRetentionPolicy(bucket_uri, expected_retention_period_in_seconds=60)

    @SkipForS3(BUCKET_LOCK_SKIP_MSG)
    def test_create_with_retention_days(self):
        bucket_name = self.MakeTempName('bucket')
        bucket_uri = boto.storage_uri('gs://%s' % bucket_name.lower(), suppress_consec_slashes=False)
        self.RunGsUtil(['mb', '--retention', '1d', suri(bucket_uri)])
        self.VerifyRetentionPolicy(bucket_uri, expected_retention_period_in_seconds=SECONDS_IN_DAY)

    @SkipForS3(BUCKET_LOCK_SKIP_MSG)
    def test_create_with_retention_months(self):
        bucket_name = self.MakeTempName('bucket')
        bucket_uri = boto.storage_uri('gs://%s' % bucket_name.lower(), suppress_consec_slashes=False)
        self.RunGsUtil(['mb', '--retention', '1m', suri(bucket_uri)])
        self.VerifyRetentionPolicy(bucket_uri, expected_retention_period_in_seconds=SECONDS_IN_MONTH)

    @SkipForS3(BUCKET_LOCK_SKIP_MSG)
    def test_create_with_retention_years(self):
        bucket_name = self.MakeTempName('bucket')
        bucket_uri = boto.storage_uri('gs://%s' % bucket_name.lower(), suppress_consec_slashes=False)
        self.RunGsUtil(['mb', '--retention', '1y', suri(bucket_uri)])
        self.VerifyRetentionPolicy(bucket_uri, expected_retention_period_in_seconds=SECONDS_IN_YEAR)

    @SkipForS3(BUCKET_LOCK_SKIP_MSG)
    def test_create_with_retention_invalid_arg(self):
        bucket_name = self.MakeTempName('bucket')
        bucket_uri = boto.storage_uri('gs://%s' % bucket_name.lower(), suppress_consec_slashes=False)
        stderr = self.RunGsUtil(['mb', '--retention', '1second', suri(bucket_uri)], expected_status=1, return_stderr=True)
        self.assertRegex(stderr, 'Incorrect retention period specified')

    def test_create_with_retention_on_s3_urls_fails(self):
        bucket_name = self.MakeTempName('bucket')
        bucket_uri = boto.storage_uri('s3://%s' % bucket_name.lower(), suppress_consec_slashes=False)
        stderr = self.RunGsUtil(['mb', '--retention', '1y', suri(bucket_uri)], expected_status=1, return_stderr=True)
        if self._use_gcloud_storage:
            self.assertIn('Features disallowed for S3: Setting Retention Period', stderr)
        else:
            self.assertRegex(stderr, 'Retention policy can only be specified for GCS buckets.')

    @SkipForXML('Public access prevention only runs on GCS JSON API.')
    def test_create_with_pap_enforced(self):
        bucket_name = self.MakeTempName('bucket')
        bucket_uri = boto.storage_uri('gs://%s' % bucket_name.lower(), suppress_consec_slashes=False)
        self.RunGsUtil(['mb', '--pap', 'enforced', suri(bucket_uri)])
        self.VerifyPublicAccessPreventionValue(bucket_uri, 'enforced')

    @SkipForXML('Public access prevention only runs on GCS JSON API.')
    def test_create_with_pap_inherited(self):
        bucket_name = self.MakeTempName('bucket')
        bucket_uri = boto.storage_uri('gs://%s' % bucket_name.lower(), suppress_consec_slashes=False)
        self.RunGsUtil(['mb', '--pap', 'inherited', suri(bucket_uri)])
        stdout = self.RunGsUtil(['publicaccessprevention', 'get', suri(bucket_uri)], return_stdout=True)
        self.assertRegex(stdout, '%s:\\s+inherited' % suri(bucket_uri))

    @SkipForXML('Public access prevention only runs on GCS JSON API.')
    def test_create_with_pap_invalid_arg(self):
        bucket_name = self.MakeTempName('bucket')
        bucket_uri = boto.storage_uri('gs://%s' % bucket_name.lower(), suppress_consec_slashes=False)
        stderr = self.RunGsUtil(['mb', '--pap', 'invalid_arg', suri(bucket_uri)], expected_status=1, return_stderr=True)
        if self._use_gcloud_storage:
            self.assertIn('Flag value not in translation map for "--pap": invalid_arg', stderr)
        else:
            self.assertRegex(stderr, 'invalid_arg is not a valid value')

    @SkipForXML('RPO flag only works for GCS JSON API.')
    def test_create_with_rpo_async_turbo(self):
        bucket_name = self.MakeTempName('bucket')
        bucket_uri = boto.storage_uri('gs://%s' % bucket_name.lower(), suppress_consec_slashes=False)
        self.RunGsUtil(['mb', '-l', 'nam4', '--rpo', 'ASYNC_TURBO', suri(bucket_uri)])
        self.VerifyCommandGet(bucket_uri, 'rpo', 'ASYNC_TURBO')

    @SkipForXML('RPO flag only works for GCS JSON API.')
    def test_create_sets_rpo_to_default(self):
        bucket_name = self.MakeTempName('bucket')
        bucket_uri = boto.storage_uri('gs://%s' % bucket_name.lower(), suppress_consec_slashes=False)
        self.RunGsUtil(['mb', '-l', 'nam4', suri(bucket_uri)])
        try:
            self.VerifyCommandGet(bucket_uri, 'rpo', 'DEFAULT')
        except AssertionError:
            self.VerifyCommandGet(bucket_uri, 'rpo', 'None')

    @SkipForXML('RPO flag only works for GCS JSON API.')
    def test_create_with_rpo_async_turbo_fails_for_regional_bucket(self):
        """Turbo replication is only meant for dual-region."""
        bucket_name = self.MakeTempName('bucket')
        bucket_uri = boto.storage_uri('gs://%s' % bucket_name.lower(), suppress_consec_slashes=False)
        stderr = self.RunGsUtil(['mb', '-l', 'us-central1', '--rpo', 'ASYNC_TURBO', suri(bucket_uri)], return_stderr=True, expected_status=1)
        self.assertIn('ASYNC_TURBO cannot be enabled on REGION bucket', stderr)

    @SkipForXML('RPO flag only works for GCS JSON API.')
    def test_create_with_rpo_incorrect_value_raises_error(self):
        bucket_name = self.MakeTempName('bucket')
        bucket_uri = boto.storage_uri('gs://%s' % bucket_name.lower(), suppress_consec_slashes=False)
        expected_status = 2 if self._use_gcloud_storage else 1
        stderr = self.RunGsUtil(['mb', '-l', 'nam4', '--rpo', 'incorrect_value', suri(bucket_uri)], return_stderr=True, expected_status=expected_status)
        if self._use_gcloud_storage:
            self.assertIn("--recovery-point-objective: Invalid choice: 'incorrect_value'", stderr)
        else:
            self.assertIn('Invalid value for --rpo. Must be one of: (ASYNC_TURBO|DEFAULT), provided: incorrect_value', stderr)

    @SkipForXML(KMS_SKIP_MSG)
    @SkipForS3(KMS_SKIP_MSG)
    def test_create_with_k_flag_not_authorized(self):
        bucket_name = self.MakeTempName('bucket')
        bucket_uri = boto.storage_uri('gs://%s' % bucket_name.lower(), suppress_consec_slashes=False)
        key = self.GetKey()
        stderr = self.RunGsUtil(['mb', '-l', testcase.KmsTestingResources.KEYRING_LOCATION, '-k', key, suri(bucket_uri)], return_stderr=True, expected_status=1)
        if self._use_gcloud_storage:
            self.assertIn('HTTPError 403: Permission denied on Cloud KMS key.', stderr)
        else:
            self.assertIn('To authorize, run:', stderr)
            self.assertIn('-k %s' % key, stderr)

    @SkipForXML(KMS_SKIP_MSG)
    @SkipForS3(KMS_SKIP_MSG)
    def test_create_with_k_flag_p_flag_not_authorized(self):
        bucket_name = self.MakeTempName('bucket')
        bucket_uri = boto.storage_uri('gs://%s' % bucket_name.lower(), suppress_consec_slashes=False)
        key = self.GetKey()
        stderr = self.RunGsUtil(['mb', '-l', testcase.KmsTestingResources.KEYRING_LOCATION, '-k', key, '-p', PopulateProjectId(), suri(bucket_uri)], return_stderr=True, expected_status=1)
        if self._use_gcloud_storage:
            self.assertIn('HTTPError 403: Permission denied on Cloud KMS key.', stderr)
        else:
            self.assertIn('To authorize, run:', stderr)
            self.assertIn('-p %s' % PopulateProjectId(), stderr)

    @SkipForXML(KMS_SKIP_MSG)
    @SkipForS3(KMS_SKIP_MSG)
    @Retry(AssertionError, tries=3, timeout_secs=1)
    def test_create_with_k_flag_authorized(self):
        bucket_name = self.MakeTempName('bucket')
        bucket_uri = boto.storage_uri('gs://%s' % bucket_name.lower(), suppress_consec_slashes=False)
        key = self.GetKey(mutable=True)
        self.RunGsUtil(['kms', 'authorize', '-k', key])
        self.RunGsUtil(['mb', '-l', testcase.KmsTestingResources.KEYRING_LOCATION, '-k', key, suri(bucket_uri)], expected_status=0)

    @SkipForXML('Custom Dual Region is not supported for the XML API.')
    @SkipForS3('Custom Dual Region is not supported for S3 buckets.')
    def test_create_with_custom_dual_regions_via_l_flag(self):
        bucket_name = self.MakeTempName('bucket')
        bucket_uri = boto.storage_uri('gs://%s' % bucket_name.lower(), suppress_consec_slashes=False)
        self.RunGsUtil(['mb', '-l', 'us-east1+us-east4', suri(bucket_uri)])
        stdout = self.RunGsUtil(['ls', '-Lb', suri(bucket_uri)], return_stdout=True)
        self.assertRegex(stdout, 'ocations:\\s*\\[\\s*.US-EAST1.,\\s*.US-EAST4')

    @SkipForXML('Custom Dual Region is not supported for the XML API.')
    @SkipForS3('Custom Dual Region is not supported for S3 buckets.')
    def test_create_with_invalid_dual_regions_via_l_flag_raises_error(self):
        bucket_name = self.MakeTempName('bucket')
        bucket_uri = boto.storage_uri('gs://%s' % bucket_name.lower(), suppress_consec_slashes=False)
        stderr = self.RunGsUtil(['mb', '-l', 'invalid_reg1+invalid_reg2', suri(bucket_uri)], return_stderr=True, expected_status=1)
        self.assertIn('The specified location constraint is not valid', stderr)

    @SkipForXML('The --placement flag only works for GCS JSON API.')
    def test_create_with_placement_flag(self):
        bucket_name = self.MakeTempName('bucket')
        bucket_uri = boto.storage_uri('gs://%s' % bucket_name.lower(), suppress_consec_slashes=False)
        self.RunGsUtil(['mb', '--placement', 'us-central1,us-west1', suri(bucket_uri)])
        stdout = self.RunGsUtil(['ls', '-Lb', suri(bucket_uri)], return_stdout=True)
        self.assertRegex(stdout, 'ocations:\\s*\\[\\s*.US-CENTRAL1.,\\s*.US-WEST1')

    @SkipForXML('The --placement flag only works for GCS JSON API.')
    def test_create_with_invalid_placement_flag_raises_error(self):
        bucket_name = self.MakeTempName('bucket')
        bucket_uri = boto.storage_uri('gs://%s' % bucket_name.lower(), suppress_consec_slashes=False)
        stderr = self.RunGsUtil(['mb', '--placement', 'invalid_reg1,invalid_reg2', suri(bucket_uri)], return_stderr=True, expected_status=1)
        self.assertRegex(stderr, '.*400.*(Invalid custom placement config|One or more unrecognized regions in dual-region, received: INVALID_REG1, INVALID_REG2).*')

    @SkipForXML('The --placement flag only works for GCS JSON API.')
    def test_create_with_incorrect_number_of_placement_values_raises_error(self):
        bucket_name = self.MakeTempName('bucket')
        bucket_uri = boto.storage_uri('gs://%s' % bucket_name.lower(), suppress_consec_slashes=False)
        expected_status = 2 if self._use_gcloud_storage else 1
        stderr = self.RunGsUtil(['mb', '--placement', 'val1,val2,val3', suri(bucket_uri)], return_stderr=True, expected_status=expected_status)
        if self._use_gcloud_storage:
            self.assertIn('--placement: too many args', stderr)
        else:
            self.assertIn('CommandException: Please specify two regions separated by comma without space. Specified: val1,val2,val3', stderr)

    @SkipForJSON('Testing XML only behavior.')
    def test_single_json_only_flag_raises_error_with_xml_api(self):
        bucket_name = self.MakeTempName('bucket')
        bucket_uri = boto.storage_uri('gs://%s' % bucket_name.lower(), suppress_consec_slashes=False)
        stderr = self.RunGsUtil(['mb', '--rpo', 'ASYNC_TURBO', suri(bucket_uri)], return_stderr=True, expected_status=1)
        self.assertIn('CommandException: The --rpo option(s) can only be used for GCS Buckets with the JSON API', stderr)

    @SkipForJSON('Testing XML only behavior.')
    def test_multiple_json_only_flags_raise_error_with_xml_api(self):
        bucket_name = self.MakeTempName('bucket')
        bucket_uri = boto.storage_uri('gs://%s' % bucket_name.lower(), suppress_consec_slashes=False)
        stderr = self.RunGsUtil(['mb', '--autoclass', '--pap', 'enabled', '--placement', 'uscentral-1,us-asia1', '--rpo', 'ASYNC_TURBO', '-b', 'on', suri(bucket_uri)], return_stderr=True, expected_status=1)
        self.assertIn('CommandException: The --autoclass, --pap, --placement, --rpo, -b option(s) can only be used for GCS Buckets with the JSON API', stderr)