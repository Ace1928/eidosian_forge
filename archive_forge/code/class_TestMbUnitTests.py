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
class TestMbUnitTests(testcase.GsUtilUnitTestCase):
    """Unit tests for gsutil mb."""

    def test_shim_translates_retention_seconds_flags(self):
        with SetBotoConfigForTest([('GSUtil', 'use_gcloud_storage', 'True'), ('GSUtil', 'hidden_shim_mode', 'dry_run')]):
            with SetEnvironmentForTest({'CLOUDSDK_CORE_PASS_CREDENTIALS_TO_GSUTIL': 'True', 'CLOUDSDK_ROOT_DIR': 'fake_dir'}):
                mock_log_handler = self.RunCommand('mb', args=['--retention', '1y', 'gs://fake-bucket'], return_log_handler=True)
                info_lines = '\n'.join(mock_log_handler.messages['info'])
                self.assertIn('Gcloud Storage Command: {} storage buckets create --retention-period 31557600s gs://fake-bucket'.format(shim_util._get_gcloud_binary_path('fake_dir')), info_lines)

    @SkipForXML('The --rpo flag only works for GCS JSON API.')
    def test_shim_translates_recovery_point_objective_flag(self):
        fake_cloudsdk_dir = 'fake_dir'
        with SetBotoConfigForTest([('GSUtil', 'use_gcloud_storage', 'True'), ('GSUtil', 'hidden_shim_mode', 'dry_run')]):
            with SetEnvironmentForTest({'CLOUDSDK_CORE_PASS_CREDENTIALS_TO_GSUTIL': 'True', 'CLOUDSDK_ROOT_DIR': fake_cloudsdk_dir}):
                mock_log_handler = self.RunCommand('mb', args=['--rpo', 'DEFAULT', 'gs://fake-bucket-1'], return_log_handler=True)
                info_lines = '\n'.join(mock_log_handler.messages['info'])
                self.assertIn('Gcloud Storage Command: {} storage buckets create --recovery-point-objective DEFAULT'.format(shim_util._get_gcloud_binary_path('fake_dir')), info_lines)