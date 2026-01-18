from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import os
import sys
from gslib.cs_api_map import ApiSelector
from gslib.exception import NO_URLS_MATCHED_TARGET
import gslib.tests.testcase as testcase
from gslib.tests.testcase.integration_testcase import SkipForS3
from gslib.tests.util import GenerationFromURI as urigen
from gslib.tests.util import ObjectToURI as suri
from gslib.tests.util import RUN_S3_TESTS
from gslib.tests.util import SetBotoConfigForTest
from gslib.tests.util import SetEnvironmentForTest
from gslib.tests.util import TEST_ENCRYPTION_KEY1
from gslib.tests.util import unittest
from gslib.utils import cat_helper
from gslib.utils import shim_util
from unittest import mock
class TestShimCatFlags(testcase.GsUtilUnitTestCase):
    """Unit tests for shimming cat flags"""

    def test_shim_translates_flags(self):
        object_uri = self.CreateObject(contents='0123456789')
        with SetBotoConfigForTest([('GSUtil', 'use_gcloud_storage', 'True'), ('GSUtil', 'hidden_shim_mode', 'dry_run')]):
            with SetEnvironmentForTest({'CLOUDSDK_CORE_PASS_CREDENTIALS_TO_GSUTIL': 'True', 'CLOUDSDK_ROOT_DIR': 'fake_dir'}):
                mock_log_handler = self.RunCommand('cat', ['-h', '-r', '2-4', suri(object_uri)], return_log_handler=True)
                info_lines = '\n'.join(mock_log_handler.messages['info'])
                self.assertIn('Gcloud Storage Command: {} storage cat -d -r 2-4 {}'.format(shim_util._get_gcloud_binary_path('fake_dir'), suri(object_uri)), info_lines)