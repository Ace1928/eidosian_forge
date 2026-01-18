from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
from datetime import datetime
import os
import posixpath
import re
import stat
import subprocess
import sys
import time
import gslib
from gslib.commands import ls
from gslib.cs_api_map import ApiSelector
from gslib.project_id import PopulateProjectId
import gslib.tests.testcase as testcase
from gslib.tests.testcase.integration_testcase import SkipForGS
from gslib.tests.testcase.integration_testcase import SkipForJSON
from gslib.tests.testcase.integration_testcase import SkipForS3
from gslib.tests.testcase.integration_testcase import SkipForXML
from gslib.tests.util import CaptureStdout
from gslib.tests.util import ObjectToURI as suri
from gslib.tests.util import RUN_S3_TESTS
from gslib.tests.util import SetBotoConfigForTest
from gslib.tests.util import SetEnvironmentForTest
from gslib.tests.util import TEST_ENCRYPTION_CONTENT1
from gslib.tests.util import TEST_ENCRYPTION_CONTENT1_CRC32C
from gslib.tests.util import TEST_ENCRYPTION_CONTENT1_MD5
from gslib.tests.util import TEST_ENCRYPTION_CONTENT2
from gslib.tests.util import TEST_ENCRYPTION_CONTENT2_CRC32C
from gslib.tests.util import TEST_ENCRYPTION_CONTENT2_MD5
from gslib.tests.util import TEST_ENCRYPTION_CONTENT3
from gslib.tests.util import TEST_ENCRYPTION_CONTENT3_CRC32C
from gslib.tests.util import TEST_ENCRYPTION_CONTENT3_MD5
from gslib.tests.util import TEST_ENCRYPTION_CONTENT4
from gslib.tests.util import TEST_ENCRYPTION_CONTENT4_CRC32C
from gslib.tests.util import TEST_ENCRYPTION_CONTENT4_MD5
from gslib.tests.util import TEST_ENCRYPTION_CONTENT5
from gslib.tests.util import TEST_ENCRYPTION_CONTENT5_CRC32C
from gslib.tests.util import TEST_ENCRYPTION_CONTENT5_MD5
from gslib.tests.util import TEST_ENCRYPTION_KEY1
from gslib.tests.util import TEST_ENCRYPTION_KEY1_SHA256_B64
from gslib.tests.util import TEST_ENCRYPTION_KEY2
from gslib.tests.util import TEST_ENCRYPTION_KEY2_SHA256_B64
from gslib.tests.util import TEST_ENCRYPTION_KEY3
from gslib.tests.util import TEST_ENCRYPTION_KEY3_SHA256_B64
from gslib.tests.util import TEST_ENCRYPTION_KEY4
from gslib.tests.util import TEST_ENCRYPTION_KEY4_SHA256_B64
from gslib.tests.util import unittest
from gslib.third_party.storage_apitools import storage_v1_messages as apitools_messages
from gslib.utils.constants import UTF8
from gslib.utils.ls_helper import PrintFullInfoAboutObject
from gslib.utils.retry_util import Retry
from gslib.utils.system_util import IS_WINDOWS
from gslib.utils import shim_util
from six import add_move, MovedModule
from six.moves import mock
class TestLsUnit(testcase.GsUtilUnitTestCase):
    """Unit tests for ls command."""

    def test_one_object_with_L_storage_class_update(self):
        """Tests the JSON storage class update time field."""
        if self.test_api == ApiSelector.XML:
            return unittest.skip('XML API has no concept of storage class update time')
        current_time = datetime(2017, 1, 2, 3, 4, 5, 6, tzinfo=None)
        obj_metadata = apitools_messages.Object(name='foo', bucket='bar', timeCreated=current_time, updated=current_time, timeStorageClassUpdated=current_time, etag='12345')
        obj_ref = mock.Mock()
        obj_ref.root_object = obj_metadata
        obj_ref.url_string = 'foo'
        with CaptureStdout() as output:
            PrintFullInfoAboutObject(obj_ref)
        output = '\n'.join(output)
        find_stor_update_re = re.compile('^\\s*Storage class update time:+(?P<stor_update_time_val>.+)$', re.MULTILINE)
        stor_update_time_match = re.search(find_stor_update_re, output)
        self.assertIsNone(stor_update_time_match)
        new_update_time = datetime(2017, 2, 3, 4, 5, 6, 7, tzinfo=None)
        obj_metadata2 = apitools_messages.Object(name='foo2', bucket='bar2', timeCreated=current_time, updated=current_time, timeStorageClassUpdated=new_update_time, etag='12345')
        obj_ref2 = mock.Mock()
        obj_ref2.root_object = obj_metadata2
        obj_ref2.url_string = 'foo2'
        with CaptureStdout() as output2:
            PrintFullInfoAboutObject(obj_ref2)
        output2 = '\n'.join(output2)
        find_time_created_re = re.compile('^\\s*Creation time:\\s+(?P<time_created_val>.+)$', re.MULTILINE)
        time_created_match = re.search(find_time_created_re, output2)
        self.assertIsNotNone(time_created_match)
        time_created = time_created_match.group('time_created_val')
        self.assertEqual(time_created, datetime.strftime(current_time, '%a, %d %b %Y %H:%M:%S GMT'))
        find_stor_update_re_2 = re.compile('^\\s*Storage class update time:+(?P<stor_update_time_val_2>.+)$', re.MULTILINE)
        stor_update_time_match_2 = re.search(find_stor_update_re_2, output2)
        self.assertIsNotNone(stor_update_time_match_2)
        stor_update_time = stor_update_time_match_2.group('stor_update_time_val_2')
        self.assertEqual(stor_update_time, datetime.strftime(new_update_time, '%a, %d %b %Y %H:%M:%S GMT'))

    @mock.patch.object(ls.LsCommand, 'WildcardIterator')
    def test_satisfies_pzs_is_displayed_if_present(self, mock_wildcard):
        bucket_uri = self.CreateBucket(bucket_name='foo')
        bucket_metadata = apitools_messages.Bucket(name='foo', satisfiesPZS=True)
        bucket_uri.root_object = bucket_metadata
        bucket_uri.url_string = 'foo'
        bucket_uri.storage_url = mock.Mock()
        mock_wildcard.return_value.IterBuckets.return_value = [bucket_uri]
        with SetBotoConfigForTest([('GSUtil', 'check_hashes', 'never')]):
            stdout = self.RunCommand('ls', ['-Lb', suri(bucket_uri)], return_stdout=True)
        self.assertRegex(stdout, 'Satisfies PZS:\t\t\tTrue')

    @mock.patch.object(ls.LsCommand, 'WildcardIterator')
    def test_placement_locations_not_displayed_for_normal_bucket(self, mock_wildcard):
        """Non custom dual region buckets should not display placement info."""
        bucket_uri = self.CreateBucket(bucket_name='foo-non-cdr')
        bucket_metadata = apitools_messages.Bucket(name='foo-non-cdr')
        bucket_uri.root_object = bucket_metadata
        bucket_uri.url_string = 'foo-non-cdr'
        bucket_uri.storage_url = mock.Mock()
        mock_wildcard.return_value.IterBuckets.return_value = [bucket_uri]
        with SetBotoConfigForTest([('GSUtil', 'check_hashes', 'never')]):
            stdout = self.RunCommand('ls', ['-Lb', suri(bucket_uri)], return_stdout=True)
        self.assertNotRegex(stdout, 'Placement locations:')

    def test_shim_translates_flags(self):
        with SetBotoConfigForTest([('GSUtil', 'use_gcloud_storage', 'True'), ('GSUtil', 'hidden_shim_mode', 'dry_run')]):
            with SetEnvironmentForTest({'CLOUDSDK_CORE_PASS_CREDENTIALS_TO_GSUTIL': 'True', 'CLOUDSDK_ROOT_DIR': 'fake_dir'}):
                mock_log_handler = self.RunCommand('ls', ['-rRlLbeah', '-p foo'], return_log_handler=True)
                self.assertIn('Gcloud Storage Command: {} storage ls --fetch-encrypted-object-hashes -r -r -l -L -b -e -a --readable-sizes --project  foo'.format(shim_util._get_gcloud_binary_path('fake_dir')), mock_log_handler.messages['info'])