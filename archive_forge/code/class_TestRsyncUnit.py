from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import os
import re
from unittest import mock
import six
from gslib import command
from gslib.commands import rsync
from gslib.project_id import PopulateProjectId
from gslib.storage_url import StorageUrlFromString
import gslib.tests.testcase as testcase
from gslib.tests.testcase.integration_testcase import SkipForGS
from gslib.tests.testcase.integration_testcase import SkipForS3
from gslib.tests.testcase.integration_testcase import SkipForXML
from gslib.tests.util import AuthorizeProjectToUseTestingKmsKey
from gslib.tests.util import TEST_ENCRYPTION_KEY_S3
from gslib.tests.util import TEST_ENCRYPTION_KEY_S3_MD5
from gslib.tests.util import BuildErrorRegex
from gslib.tests.util import ObjectToURI as suri
from gslib.tests.util import ORPHANED_FILE
from gslib.tests.util import POSIX_GID_ERROR
from gslib.tests.util import POSIX_INSUFFICIENT_ACCESS_ERROR
from gslib.tests.util import POSIX_MODE_ERROR
from gslib.tests.util import POSIX_UID_ERROR
from gslib.tests.util import SequentialAndParallelTransfer
from gslib.tests.util import SetBotoConfigForTest
from gslib.tests.util import SetEnvironmentForTest
from gslib.tests.util import TailSet
from gslib.tests.util import unittest
from gslib.utils.boto_util import UsingCrcmodExtension
from gslib.utils.hashing_helper import SLOW_CRCMOD_RSYNC_WARNING
from gslib.utils.posix_util import ConvertDatetimeToPOSIX
from gslib.utils.posix_util import GID_ATTR
from gslib.utils.posix_util import MODE_ATTR
from gslib.utils.posix_util import MTIME_ATTR
from gslib.utils.posix_util import NA_TIME
from gslib.utils.posix_util import UID_ATTR
from gslib.utils.retry_util import Retry
from gslib.utils.system_util import IS_OSX
from gslib.utils.system_util import IS_WINDOWS
from gslib.utils import shim_util
class TestRsyncUnit(testcase.GsUtilUnitTestCase):
    """Unit tests for methods in the commands.rsync module."""

    def testUrlEncodeAndDecode(self):
        decoded_url = 'gs://bkt/space fslash/plus+tilde~unicodeeè'
        encoded_url = 'gs%3A%2F%2Fbkt%2Fspace+fslash%2Fplus%2Btilde~unicodee%C3%A8'
        self.assertEqual(rsync._EncodeUrl(six.ensure_text(decoded_url)), six.ensure_str(encoded_url))
        self.assertEqual(rsync._DecodeUrl(six.ensure_str(encoded_url)), six.ensure_text(decoded_url))

    @mock.patch('gslib.utils.copy_helper.TriggerReauthForDestinationProviderIfNecessary')
    @mock.patch('gslib.command.Command._GetProcessAndThreadCount')
    @mock.patch('gslib.command.Command.Apply', new=mock.Mock(spec=command.Command.Apply))
    def testRsyncTriggersReauth(self, mock_get_process_and_thread_count, mock_trigger_reauth):
        path = self.CreateTempDir()
        bucket_uri = self.CreateBucket()
        mock_get_process_and_thread_count.return_value = (2, 3)
        self.RunCommand('rsync', [path, suri(bucket_uri)])
        mock_trigger_reauth.assert_called_once_with(StorageUrlFromString(suri(bucket_uri)), mock.ANY, worker_count=6)
        mock_get_process_and_thread_count.assert_called_once_with(process_count=None, thread_count=None, parallel_operations_override=command.Command.ParallelOverrideReason.SPEED, print_macos_warning=False)

    def testShimTranslatesFlags(self):
        bucket_uri = self.CreateBucket()
        fpath = self.CreateTempDir()
        with SetBotoConfigForTest([('GSUtil', 'use_gcloud_storage', 'True'), ('GSUtil', 'hidden_shim_mode', 'dry_run')]):
            with SetEnvironmentForTest({'CLOUDSDK_CORE_PASS_CREDENTIALS_TO_GSUTIL': 'True', 'CLOUDSDK_ROOT_DIR': 'fake_dir'}):
                mock_log_handler = self.RunCommand('rsync', ['-a', 'public-read', '-P', suri(bucket_uri), fpath], return_log_handler=True)
                info_lines = '\n'.join(mock_log_handler.messages['info'])
                self.assertIn('Gcloud Storage Command: {} storage rsync --no-ignore-symlinks --predefined-acl publicRead --preserve-posix {} {}'.format(shim_util._get_gcloud_binary_path('fake_dir'), suri(bucket_uri), fpath), info_lines)
                warn_lines = '\n'.join(mock_log_handler.messages['warning'])
                self.assertIn('By default, gsutil copies file symlinks', warn_lines)
                self.assertIn('For preserving POSIX with rsync downloads', warn_lines)