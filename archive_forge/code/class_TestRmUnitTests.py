from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import os
import re
import sys
from unittest import mock
from gslib.exception import NO_URLS_MATCHED_PREFIX
from gslib.exception import NO_URLS_MATCHED_TARGET
import gslib.tests.testcase as testcase
from gslib.tests.testcase.base import MAX_BUCKET_LENGTH
from gslib.tests.testcase.integration_testcase import SkipForS3
import gslib.tests.util as util
from gslib.tests.util import GenerationFromURI as urigen
from gslib.tests.util import ObjectToURI as suri
from gslib.tests.util import SetBotoConfigForTest
from gslib.tests.util import SetEnvironmentForTest
from gslib.utils import shim_util
from gslib.utils.retry_util import Retry
class TestRmUnitTests(testcase.GsUtilUnitTestCase):
    """Unit tests for gsutil rm."""

    def test_shim_translates_flags(self):
        bucket_uri = self.CreateBucket()
        with SetBotoConfigForTest([('GSUtil', 'use_gcloud_storage', 'True'), ('GSUtil', 'hidden_shim_mode', 'dry_run')]):
            with SetEnvironmentForTest({'CLOUDSDK_CORE_PASS_CREDENTIALS_TO_GSUTIL': 'True', 'CLOUDSDK_ROOT_DIR': 'fake_dir'}):
                mock_log_handler = self.RunCommand('rm', ['-r', '-R', '-a', '-f', suri(bucket_uri)], return_log_handler=True)
                info_lines = '\n'.join(mock_log_handler.messages['info'])
                self.assertIn('Gcloud Storage Command: {} storage rm -r -r -a --continue-on-error {}'.format(shim_util._get_gcloud_binary_path('fake_dir'), suri(bucket_uri)), info_lines)

    @mock.patch.object(sys, 'stdin')
    def test_shim_translates_stdin_flag(self, mock_stdin):
        bucket_uri = self.CreateBucket()
        object_uri = self.CreateObject(bucket_uri, 'foo', 'abcd')
        mock_stdin.__iter__.return_value = [suri(object_uri)]
        with SetBotoConfigForTest([('GSUtil', 'use_gcloud_storage', 'True'), ('GSUtil', 'hidden_shim_mode', 'dry_run')]):
            with SetEnvironmentForTest({'CLOUDSDK_CORE_PASS_CREDENTIALS_TO_GSUTIL': 'True', 'CLOUDSDK_ROOT_DIR': 'fake_dir'}):
                mock_log_handler = self.RunCommand('rm', ['-I'], return_log_handler=True)
                info_lines = '\n'.join(mock_log_handler.messages['info'])
                self.assertIn('Gcloud Storage Command: {} storage rm -I'.format(shim_util._get_gcloud_binary_path('fake_dir')), info_lines)