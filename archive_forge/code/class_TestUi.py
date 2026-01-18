from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import os
import pickle
import crcmod
import six
from six.moves import queue as Queue
from gslib.cs_api_map import ApiSelector
from gslib.parallel_tracker_file import ObjectFromTracker
from gslib.parallel_tracker_file import WriteParallelUploadTrackerFile
from gslib.storage_url import StorageUrlFromString
import gslib.tests.testcase as testcase
from gslib.tests.testcase.integration_testcase import SkipForS3
from gslib.tests.util import HaltingCopyCallbackHandler
from gslib.tests.util import HaltOneComponentCopyCallbackHandler
from gslib.tests.util import ObjectToURI as suri
from gslib.tests.util import SetBotoConfigForTest
from gslib.tests.util import TailSet
from gslib.tests.util import TEST_ENCRYPTION_KEY1
from gslib.tests.util import TEST_ENCRYPTION_KEY2
from gslib.tests.util import unittest
from gslib.thread_message import FileMessage
from gslib.thread_message import FinalMessage
from gslib.thread_message import MetadataMessage
from gslib.thread_message import ProducerThreadMessage
from gslib.thread_message import ProgressMessage
from gslib.thread_message import SeekAheadMessage
from gslib.tracker_file import DeleteTrackerFile
from gslib.tracker_file import GetSlicedDownloadTrackerFilePaths
from gslib.tracker_file import GetTrackerFilePath
from gslib.tracker_file import TrackerFileType
from gslib.ui_controller import BytesToFixedWidthString
from gslib.ui_controller import DataManager
from gslib.ui_controller import MainThreadUIQueue
from gslib.ui_controller import MetadataManager
from gslib.ui_controller import UIController
from gslib.ui_controller import UIThread
from gslib.utils.boto_util import UsingCrcmodExtension
from gslib.utils.constants import START_CALLBACK_PER_BYTES
from gslib.utils.constants import UTF8
from gslib.utils.copy_helper import PARALLEL_UPLOAD_STATIC_SALT
from gslib.utils.copy_helper import PARALLEL_UPLOAD_TEMP_NAMESPACE
from gslib.utils.hashing_helper import GetMd5
from gslib.utils.parallelism_framework_util import PutToQueueWithTimeout
from gslib.utils.parallelism_framework_util import ZERO_TASKS_TO_DO_ARGUMENT
from gslib.utils.retry_util import Retry
from gslib.utils.unit_util import HumanReadableWithDecimalPlaces
from gslib.utils.unit_util import MakeHumanReadable
from gslib.utils.unit_util import ONE_KIB
class TestUi(testcase.GsUtilIntegrationTestCase):
    """Integration tests for UI functions."""

    def test_ui_download_single_objects_with_m_flag(self):
        """Tests UI for a single object download with the -m flag enabled.

    This test indirectly tests the correctness of ProducerThreadMessage in the
    UIController.
    """
        bucket_uri = self.CreateBucket()
        file_contents = b'd' * DOWNLOAD_SIZE
        object_uri = self.CreateObject(bucket_uri=bucket_uri, object_name='foo', contents=file_contents)
        fpath = self.CreateTempFile()
        stderr = self.RunGsUtil(['-m', 'cp', suri(object_uri), fpath], return_stderr=True)
        CheckUiOutputWithMFlag(self, stderr, 1, total_size=DOWNLOAD_SIZE)

    def test_ui_download_single_objects_with_no_m_flag(self):
        """Tests UI for a single object download with the -m flag not enabled.

    The UI should behave differently from the -m flag option because in the
    latter we have a ProducerThreadMessage that allows us to know our progress
    percentage and total number of files.
    """
        bucket_uri = self.CreateBucket()
        file_contents = b'd' * DOWNLOAD_SIZE
        object_uri = self.CreateObject(bucket_uri=bucket_uri, object_name='foo', contents=file_contents)
        fpath = self.CreateTempFile()
        stderr = self.RunGsUtil(['cp', suri(object_uri), fpath], return_stderr=True)
        CheckUiOutputWithNoMFlag(self, stderr, 1, total_size=DOWNLOAD_SIZE)

    def test_ui_upload_single_object_with_m_flag(self):
        """Tests UI for a single object upload with -m flag enabled.

    This test indirectly tests the correctness of ProducerThreadMessage in the
    UIController.
    """
        bucket_uri = self.CreateBucket()
        file_contents = b'u' * UPLOAD_SIZE
        fpath = self.CreateTempFile(file_name='sample-file.txt', contents=file_contents)
        stderr = self.RunGsUtil(['-m', 'cp', suri(fpath), suri(bucket_uri)], return_stderr=True)
        CheckUiOutputWithMFlag(self, stderr, 1, total_size=UPLOAD_SIZE)

    def test_ui_upload_single_object_with_no_m_flag(self):
        """Tests UI for a single object upload with -m flag not enabled.

    The UI should behave differently from the -m flag option because in the
    latter we have a ProducerThreadMessage that allows us to know our progress
    percentage and total number of files.
    """
        bucket_uri = self.CreateBucket()
        file_contents = b'u' * UPLOAD_SIZE
        fpath = self.CreateTempFile(file_name='sample-file.txt', contents=file_contents)
        stderr = self.RunGsUtil(['cp', suri(fpath), suri(bucket_uri)], return_stderr=True)
        CheckUiOutputWithNoMFlag(self, stderr, 1, total_size=UPLOAD_SIZE)

    def test_ui_download_multiple_objects_with_m_flag(self):
        """Tests UI for a multiple object download with the -m flag enabled.

    This test indirectly tests the correctness of ProducerThreadMessage in the
    UIController.
    """
        bucket_uri = self.CreateBucket()
        num_objects = 7
        argument_list = ['-m', 'cp']
        total_size = 0
        for i in range(num_objects):
            file_size = DOWNLOAD_SIZE // 3
            file_contents = b'd' * file_size
            object_uri = self.CreateObject(bucket_uri=bucket_uri, object_name='foo' + str(i), contents=file_contents)
            total_size += file_size
            argument_list.append(suri(object_uri))
        fpath = self.CreateTempDir()
        argument_list.append(fpath)
        stderr = self.RunGsUtil(argument_list, return_stderr=True)
        CheckUiOutputWithMFlag(self, stderr, num_objects, total_size=total_size)

    def test_ui_download_multiple_objects_with_no_m_flag(self):
        """Tests UI for a multiple object download with the -m flag not enabled.

    The UI should behave differently from the -m flag option because in the
    latter we have a ProducerThreadMessage that allows us to know our progress
    percentage and total number of files.
    """
        bucket_uri = self.CreateBucket()
        num_objects = 7
        argument_list = ['cp']
        total_size = 0
        for i in range(num_objects):
            file_size = DOWNLOAD_SIZE // 3
            file_contents = b'd' * file_size
            object_uri = self.CreateObject(bucket_uri=bucket_uri, object_name='foo' + str(i), contents=file_contents)
            total_size += file_size
            argument_list.append(suri(object_uri))
        fpath = self.CreateTempDir()
        argument_list.append(fpath)
        stderr = self.RunGsUtil(argument_list, return_stderr=True)
        CheckUiOutputWithNoMFlag(self, stderr, num_objects, total_size=total_size)

    def test_ui_upload_mutliple_objects_with_m_flag(self):
        """Tests UI for a multiple object upload with -m flag enabled.

    This test indirectly tests the correctness of ProducerThreadMessage in the
    UIController.
    """
        bucket_uri = self.CreateBucket()
        num_objects = 7
        argument_list = ['-m', 'cp']
        total_size = 0
        for i in range(num_objects):
            file_size = UPLOAD_SIZE // 3
            file_contents = b'u' * file_size
            fpath = self.CreateTempFile(file_name='foo' + str(i), contents=file_contents)
            total_size += file_size
            argument_list.append(suri(fpath))
        argument_list.append(suri(bucket_uri))
        stderr = self.RunGsUtil(argument_list, return_stderr=True)
        CheckUiOutputWithMFlag(self, stderr, num_objects, total_size=total_size)

    def test_ui_upload_mutliple_objects_with_no_m_flag(self):
        """Tests UI for a multiple object upload with -m flag not enabled.

    The UI should behave differently from the -m flag option because in the
    latter we have a ProducerThreadMessage that allows us to know our progress
    percentage and total number of files.
    """
        bucket_uri = self.CreateBucket()
        num_objects = 7
        argument_list = ['cp']
        total_size = 0
        for i in range(num_objects):
            file_size = UPLOAD_SIZE // 3
            file_contents = b'u' * file_size
            fpath = self.CreateTempFile(file_name='foo' + str(i), contents=file_contents)
            total_size += file_size
            argument_list.append(suri(fpath))
        argument_list.append(suri(bucket_uri))
        stderr = self.RunGsUtil(argument_list, return_stderr=True)
        CheckUiOutputWithNoMFlag(self, stderr, num_objects, total_size=total_size)

    @SkipForS3('No resumable upload support for S3.')
    def test_ui_resumable_upload_break_with_m_flag(self):
        """Tests UI for upload resumed after a connection break with -m flag.

    This was adapted from test_cp_resumable_upload_break.
    """
        bucket_uri = self.CreateBucket()
        fpath = self.CreateTempFile(contents=b'a' * HALT_SIZE)
        boto_config_for_test = [('GSUtil', 'resumable_threshold', str(ONE_KIB)), ('GSUtil', 'parallel_composite_upload_component_size', str(ONE_KIB))]
        test_callback_file = self.CreateTempFile(contents=pickle.dumps(HaltingCopyCallbackHandler(True, 5)))
        with SetBotoConfigForTest(boto_config_for_test):
            stderr = self.RunGsUtil(['-m', 'cp', '--testcallbackfile', test_callback_file, fpath, suri(bucket_uri)], expected_status=1, return_stderr=True)
            self.assertIn('Artifically halting upload', stderr)
            CheckBrokenUiOutputWithMFlag(self, stderr, 1, total_size=HALT_SIZE)
            stderr = self.RunGsUtil(['-m', 'cp', fpath, suri(bucket_uri)], return_stderr=True)
            self.assertIn('Resuming upload', stderr)
            CheckUiOutputWithMFlag(self, stderr, 1, total_size=HALT_SIZE)

    @SkipForS3('No resumable upload support for S3.')
    def test_ui_resumable_upload_break_with_no_m_flag(self):
        """Tests UI for upload resumed after a connection break with no -m flag.

    This was adapted from test_cp_resumable_upload_break.
    """
        bucket_uri = self.CreateBucket()
        fpath = self.CreateTempFile(contents=b'a' * HALT_SIZE)
        boto_config_for_test = [('GSUtil', 'resumable_threshold', str(ONE_KIB)), ('GSUtil', 'parallel_composite_upload_component_size', str(ONE_KIB))]
        test_callback_file = self.CreateTempFile(contents=pickle.dumps(HaltingCopyCallbackHandler(True, 5)))
        with SetBotoConfigForTest(boto_config_for_test):
            stderr = self.RunGsUtil(['cp', '--testcallbackfile', test_callback_file, fpath, suri(bucket_uri)], expected_status=1, return_stderr=True)
            self.assertIn('Artifically halting upload', stderr)
            CheckBrokenUiOutputWithNoMFlag(self, stderr, 1, total_size=HALT_SIZE)
            stderr = self.RunGsUtil(['cp', fpath, suri(bucket_uri)], return_stderr=True)
            self.assertIn('Resuming upload', stderr)
            CheckUiOutputWithNoMFlag(self, stderr, 1, total_size=HALT_SIZE)

    def _test_ui_resumable_download_break_helper(self, boto_config, gsutil_flags=None):
        """Helper function for testing UI on a resumable download break.

    This was adapted from _test_cp_resumable_download_break_helper.

    Args:
      boto_config: List of boto configuration tuples for use with
          SetBotoConfigForTest.
      gsutil_flags: List of flags to run gsutil with, or None.
    """
        if not gsutil_flags:
            gsutil_flags = []
        bucket_uri = self.CreateBucket()
        file_contents = b'a' * HALT_SIZE
        object_uri = self.CreateObject(bucket_uri=bucket_uri, object_name='foo', contents=file_contents)
        fpath = self.CreateTempFile()
        test_callback_file = self.CreateTempFile(contents=pickle.dumps(HaltingCopyCallbackHandler(False, 5)))
        with SetBotoConfigForTest(boto_config):
            gsutil_args = gsutil_flags + ['cp', '--testcallbackfile', test_callback_file, suri(object_uri), fpath]
            stderr = self.RunGsUtil(gsutil_args, expected_status=1, return_stderr=True)
            self.assertIn('Artifically halting download.', stderr)
            if '-q' not in gsutil_flags:
                if '-m' in gsutil_flags:
                    CheckBrokenUiOutputWithMFlag(self, stderr, 1, total_size=HALT_SIZE)
                else:
                    CheckBrokenUiOutputWithNoMFlag(self, stderr, 1, total_size=HALT_SIZE)
            tracker_filename = GetTrackerFilePath(StorageUrlFromString(fpath), TrackerFileType.DOWNLOAD, self.test_api)
            self.assertTrue(os.path.isfile(tracker_filename))
            gsutil_args = gsutil_flags + ['cp', suri(object_uri), fpath]
            stderr = self.RunGsUtil(gsutil_args, return_stderr=True)
            if '-q' not in gsutil_args:
                self.assertIn('Resuming download', stderr)
        with open(fpath, 'rb') as f:
            self.assertEqual(f.read(), file_contents, 'File contents differ')
        if '-q' in gsutil_flags:
            self.assertEqual('', stderr)
        elif '-m' in gsutil_flags:
            CheckUiOutputWithMFlag(self, stderr, 1, total_size=HALT_SIZE)
        else:
            CheckUiOutputWithNoMFlag(self, stderr, 1, total_size=HALT_SIZE)

    def test_ui_resumable_download_break_with_m_flag(self):
        """Tests UI on a resumable download break with -m flag.

    This was adapted from test_cp_resumable_download_break.
    """
        self._test_ui_resumable_download_break_helper([('GSUtil', 'resumable_threshold', str(ONE_KIB))], gsutil_flags=['-m'])

    def test_ui_resumable_download_break_with_no_m_flag(self):
        """Tests UI on a resumable download break with no -m flag.

    This was adapted from test_cp_resumable_download_break.
    """
        self._test_ui_resumable_download_break_helper([('GSUtil', 'resumable_threshold', str(ONE_KIB))])

    def test_ui_resumable_download_break_with_q_flag(self):
        """Tests UI on a resumable download break with -q flag but no -m flag.

    This was adapted from test_cp_resumable_download_break, and the UI output
    should be empty.
    """
        self._test_ui_resumable_download_break_helper([('GSUtil', 'resumable_threshold', str(ONE_KIB))], gsutil_flags=['-q'])

    def test_ui_resumable_download_break_with_q_and_m_flags(self):
        """Tests UI on a resumable download break with -q and -m flags.

    This was adapted from test_cp_resumable_download_break, and the UI output
    should be empty.
    """
        self._test_ui_resumable_download_break_helper([('GSUtil', 'resumable_threshold', str(ONE_KIB))], gsutil_flags=['-m', '-q'])

    def _test_ui_composite_upload_resume_helper(self, gsutil_flags=None):
        """Helps testing UI on a resumable upload with finished components.

    Args:
      gsutil_flags: List of flags to run gsutil with, or None.
    """
        if not gsutil_flags:
            gsutil_flags = []
        bucket_uri = self.CreateBucket()
        dst_url = StorageUrlFromString(suri(bucket_uri, 'foo'))
        file_contents = b'foobar'
        file_name = 'foobar'
        source_file = self.CreateTempFile(contents=file_contents, file_name=file_name)
        src_url = StorageUrlFromString(source_file)
        tracker_file_name = GetTrackerFilePath(dst_url, TrackerFileType.PARALLEL_UPLOAD, self.test_api, src_url)
        tracker_prefix = '123'
        encoded_name = (PARALLEL_UPLOAD_STATIC_SALT + source_file).encode(UTF8)
        content_md5 = GetMd5()
        content_md5.update(encoded_name)
        digest = content_md5.hexdigest()
        component_object_name = tracker_prefix + PARALLEL_UPLOAD_TEMP_NAMESPACE + digest + '_0'
        component_size = 3
        object_uri = self.CreateObject(bucket_uri=bucket_uri, object_name=component_object_name, contents=file_contents[:component_size])
        existing_component = ObjectFromTracker(component_object_name, str(object_uri.generation))
        existing_components = [existing_component]
        WriteParallelUploadTrackerFile(tracker_file_name, tracker_prefix, existing_components)
        try:
            with SetBotoConfigForTest([('GSUtil', 'parallel_composite_upload_threshold', '1'), ('GSUtil', 'parallel_composite_upload_component_size', str(component_size))]):
                gsutil_args = gsutil_flags + ['cp', source_file, suri(bucket_uri, 'foo')]
                stderr = self.RunGsUtil(gsutil_args, return_stderr=True)
                self.assertIn('Found 1 existing temporary components to reuse.', stderr)
                self.assertFalse(os.path.exists(tracker_file_name), 'Tracker file %s should have been deleted.' % tracker_file_name)
                read_contents = self.RunGsUtil(['cat', suri(bucket_uri, 'foo')], return_stdout=True)
                self.assertEqual(read_contents.encode(UTF8), file_contents)
                if '-m' in gsutil_flags:
                    CheckUiOutputWithMFlag(self, stderr, 1, total_size=len(file_contents))
                else:
                    CheckUiOutputWithNoMFlag(self, stderr, 1, total_size=len(file_contents))
        finally:
            DeleteTrackerFile(tracker_file_name)

    @SkipForS3('No resumable upload support for S3.')
    def test_ui_composite_upload_resume_with_m_flag(self):
        """Tests UI on a resumable upload with finished components and -m flag."""
        self._test_ui_composite_upload_resume_helper(gsutil_flags=['-m'])

    @SkipForS3('No resumable upload support for S3.')
    def test_ui_composite_upload_resume_with_no_m_flag(self):
        """Tests UI on a resumable upload with finished components and no -m flag.
    """
        self._test_ui_composite_upload_resume_helper()

    @unittest.skipUnless(UsingCrcmodExtension(), 'Sliced download requires fast crcmod.')
    @SkipForS3('No sliced download support for S3.')
    def _test_ui_sliced_download_partial_resume_helper(self, gsutil_flags=None):
        """Helps testing UI for sliced download with some finished components.

    This was adapted from test_sliced_download_partial_resume_helper.

    Args:
      gsutil_flags: List of flags to run gsutil with, or None.
    """
        if not gsutil_flags:
            gsutil_flags = []
        bucket_uri = self.CreateBucket()
        object_uri = self.CreateObject(bucket_uri=bucket_uri, object_name='foo', contents=b'abc' * HALT_SIZE)
        fpath = self.CreateTempFile()
        test_callback_file = self.CreateTempFile(contents=pickle.dumps(HaltOneComponentCopyCallbackHandler(5)))
        boto_config_for_test = [('GSUtil', 'resumable_threshold', str(HALT_SIZE)), ('GSUtil', 'sliced_object_download_threshold', str(HALT_SIZE)), ('GSUtil', 'sliced_object_download_max_components', '3')]
        with SetBotoConfigForTest(boto_config_for_test):
            gsutil_args = gsutil_flags + ['cp', '--testcallbackfile', test_callback_file, suri(object_uri), suri(fpath)]
            stderr = self.RunGsUtil(gsutil_args, return_stderr=True, expected_status=1)
            if '-m' in gsutil_args:
                CheckBrokenUiOutputWithMFlag(self, stderr, 1, total_size=len('abc') * HALT_SIZE)
            else:
                CheckBrokenUiOutputWithNoMFlag(self, stderr, 1, total_size=len('abc') * HALT_SIZE)
            tracker_filenames = GetSlicedDownloadTrackerFilePaths(StorageUrlFromString(fpath), self.test_api)
            for tracker_filename in tracker_filenames:
                self.assertTrue(os.path.isfile(tracker_filename))
            gsutil_args = gsutil_flags + ['cp', suri(object_uri), fpath]
            stderr = self.RunGsUtil(gsutil_args, return_stderr=True)
            self.assertIn('Resuming download', stderr)
            self.assertIn('Download already complete', stderr)
            tracker_filenames = GetSlicedDownloadTrackerFilePaths(StorageUrlFromString(fpath), self.test_api)
            for tracker_filename in tracker_filenames:
                self.assertFalse(os.path.isfile(tracker_filename))
            with open(fpath, 'r') as f:
                self.assertEqual(f.read(), 'abc' * HALT_SIZE, 'File contents differ')
            if '-m' in gsutil_args:
                CheckUiOutputWithMFlag(self, stderr, 1, total_size=len('abc') * HALT_SIZE)
            else:
                CheckUiOutputWithNoMFlag(self, stderr, 1, total_size=len('abc') * HALT_SIZE)

    @SkipForS3('No resumable upload support for S3.')
    def test_ui_sliced_download_partial_resume_helper_with_m_flag(self):
        """Tests UI on a resumable download with finished components and -m flag.
    """
        self._test_ui_sliced_download_partial_resume_helper(gsutil_flags=['-m'])

    @SkipForS3('No resumable upload support for S3.')
    def _test_ui_sliced_download_partial_resume_helper_with_no_m_flag(self):
        """Tests UI on a resumable upload with finished components and no -m flag.
    """
        self._test_ui_sliced_download_partial_resume_helper()

    def test_ui_hash_mutliple_objects_with_no_m_flag(self):
        """Tests UI for a multiple object hashing with no -m flag enabled.

    This test indirectly tests the correctness of ProducerThreadMessage in the
    UIController.
    """
        num_objects = 7
        argument_list = ['hash']
        total_size = 0
        for i in range(num_objects):
            file_size = UPLOAD_SIZE // 3
            file_contents = b'u' * file_size
            fpath = self.CreateTempFile(file_name='foo' + str(i), contents=file_contents)
            total_size += file_size
            argument_list.append(suri(fpath))
        stderr = self.RunGsUtil(argument_list, return_stderr=True)
        CheckUiOutputWithNoMFlag(self, stderr, num_objects, total_size)

    def test_ui_rewrite_with_m_flag(self):
        """Tests UI output for rewrite and -m flag.

    Adapted from test_rewrite_stdin_args.
    """
        if self.test_api == ApiSelector.XML:
            return unittest.skip('Rewrite API is only supported in JSON.')
        object_uri = self.CreateObject(contents=b'bar', encryption_key=TEST_ENCRYPTION_KEY1)
        stdin_arg = suri(object_uri)
        boto_config_for_test = [('GSUtil', 'encryption_key', TEST_ENCRYPTION_KEY2), ('GSUtil', 'decryption_key1', TEST_ENCRYPTION_KEY1)]
        with SetBotoConfigForTest(boto_config_for_test):
            stderr = self.RunGsUtil(['-m', 'rewrite', '-k', '-I'], stdin=stdin_arg, return_stderr=True)
        self.AssertObjectUsesCSEK(stdin_arg, TEST_ENCRYPTION_KEY2)
        num_objects = 1
        total_size = len(b'bar')
        CheckUiOutputWithMFlag(self, stderr, num_objects, total_size)

    def test_ui_rewrite_with_no_m_flag(self):
        """Tests UI output for rewrite and -m flag not enabled.

    Adapted from test_rewrite_stdin_args.
    """
        if self.test_api == ApiSelector.XML:
            return unittest.skip('Rewrite API is only supported in JSON.')
        object_uri = self.CreateObject(contents=b'bar', encryption_key=TEST_ENCRYPTION_KEY1)
        stdin_arg = suri(object_uri)
        boto_config_for_test = [('GSUtil', 'encryption_key', TEST_ENCRYPTION_KEY2), ('GSUtil', 'decryption_key1', TEST_ENCRYPTION_KEY1)]
        with SetBotoConfigForTest(boto_config_for_test):
            stderr = self.RunGsUtil(['rewrite', '-k', '-I'], stdin=stdin_arg, return_stderr=True)
        self.AssertObjectUsesCSEK(stdin_arg, TEST_ENCRYPTION_KEY2)
        num_objects = 1
        total_size = len(b'bar')
        CheckUiOutputWithNoMFlag(self, stderr, num_objects, total_size)

    def test_ui_setmeta_with_m_flag(self):
        """Tests a recursive setmeta command with m flag has expected UI output.

    Adapted from test_recursion_works on test_setmeta.
    """
        bucket_uri = self.CreateBucket()
        object1_uri = self.CreateObject(bucket_uri=bucket_uri, contents=b'foo')
        object2_uri = self.CreateObject(bucket_uri=bucket_uri, contents=b'foo')
        stderr = self.RunGsUtil(['-m', 'setmeta', '-h', 'content-type:footype', suri(object1_uri), suri(object2_uri)], return_stderr=True)
        for obj_uri in [object1_uri, object2_uri]:
            stdout = self.RunGsUtil(['stat', suri(obj_uri)], return_stdout=True)
            self.assertIn('footype', stdout)
        CheckUiOutputWithMFlag(self, stderr, 2, metadata=True)

    def test_ui_setmeta_with_no_m_flag(self):
        """Tests a recursive setmeta command with no m flag has expected UI output.

    Adapted from test_recursion_works on test_setmeta.
    """
        bucket_uri = self.CreateBucket()
        object1_uri = self.CreateObject(bucket_uri=bucket_uri, contents=b'foo')
        object2_uri = self.CreateObject(bucket_uri=bucket_uri, contents=b'foo')
        stderr = self.RunGsUtil(['setmeta', '-h', 'content-type:footype', suri(object1_uri), suri(object2_uri)], return_stderr=True)
        for obj_uri in [object1_uri, object2_uri]:
            stdout = self.RunGsUtil(['stat', suri(obj_uri)], return_stdout=True)
            self.assertIn('footype', stdout)
        CheckUiOutputWithNoMFlag(self, stderr, 2, metadata=True)

    def test_ui_acl_with_m_flag(self):
        """Tests UI output for an ACL command with m flag enabled.

    Adapted from test_set_valid_acl_object.
    """
        get_acl_prefix = ['-m', 'acl', 'get']
        set_acl_prefix = ['-m', 'acl', 'set']
        obj_uri = suri(self.CreateObject(contents=b'foo'))
        acl_string = self.RunGsUtil(get_acl_prefix + [obj_uri], return_stdout=True)
        inpath = self.CreateTempFile(contents=acl_string.encode(UTF8))
        stderr = self.RunGsUtil(set_acl_prefix + ['public-read', obj_uri], return_stderr=True)
        CheckUiOutputWithMFlag(self, stderr, 1, metadata=True)
        acl_string2 = self.RunGsUtil(get_acl_prefix + [obj_uri], return_stdout=True)
        stderr = self.RunGsUtil(set_acl_prefix + [inpath, obj_uri], return_stderr=True)
        CheckUiOutputWithMFlag(self, stderr, 1, metadata=True)
        acl_string3 = self.RunGsUtil(get_acl_prefix + [obj_uri], return_stdout=True)
        self.assertNotEqual(acl_string, acl_string2)
        self.assertEqual(acl_string, acl_string3)

    def test_ui_acl_with_no_m_flag(self):
        """Tests UI output for an ACL command with m flag not enabled.

    Adapted from test_set_valid_acl_object.
    """
        get_acl_prefix = ['acl', 'get']
        set_acl_prefix = ['acl', 'set']
        obj_uri = suri(self.CreateObject(contents=b'foo'))
        acl_string = self.RunGsUtil(get_acl_prefix + [obj_uri], return_stdout=True)
        inpath = self.CreateTempFile(contents=acl_string.encode(UTF8))
        stderr = self.RunGsUtil(set_acl_prefix + ['public-read', obj_uri], return_stderr=True)
        CheckUiOutputWithNoMFlag(self, stderr, 1, metadata=True)
        acl_string2 = self.RunGsUtil(get_acl_prefix + [obj_uri], return_stdout=True)
        stderr = self.RunGsUtil(set_acl_prefix + [inpath, obj_uri], return_stderr=True)
        CheckUiOutputWithNoMFlag(self, stderr, 1, metadata=True)
        acl_string3 = self.RunGsUtil(get_acl_prefix + [obj_uri], return_stdout=True)
        self.assertNotEqual(acl_string, acl_string2)
        self.assertEqual(acl_string, acl_string3)

    def _test_ui_rsync_bucket_to_bucket_helper(self, gsutil_flags=None):
        """Helper class to test UI output for rsync command.

    Args:
      gsutil_flags: List of flags to run gsutil with, or None.

    Adapted from test_bucket_to_bucket in test_rsync.
    """
        if not gsutil_flags:
            gsutil_flags = []
        bucket1_uri = self.CreateBucket()
        bucket2_uri = self.CreateBucket()
        self.CreateObject(bucket_uri=bucket1_uri, object_name='obj1', contents=b'obj1')
        self.CreateObject(bucket_uri=bucket1_uri, object_name='.obj2', contents=b'.obj2', mtime=10)
        self.CreateObject(bucket_uri=bucket1_uri, object_name='subdir/obj3', contents=b'subdir/obj3')
        self.CreateObject(bucket_uri=bucket1_uri, object_name='obj6', contents=b'obj6_', mtime=100)
        self.CreateObject(bucket_uri=bucket2_uri, object_name='.obj2', contents=b'.OBJ2')
        self.CreateObject(bucket_uri=bucket2_uri, object_name='obj4', contents=b'obj4')
        self.CreateObject(bucket_uri=bucket2_uri, object_name='subdir/obj5', contents=b'subdir/obj5')
        self.CreateObject(bucket_uri=bucket2_uri, object_name='obj6', contents=b'obj6', mtime=100)

        @Retry(AssertionError, tries=3, timeout_secs=1)
        def _Check1():
            """Tests rsync works as expected."""
            gsutil_args = gsutil_flags + ['rsync', suri(bucket1_uri), suri(bucket2_uri)]
            stderr = self.RunGsUtil(gsutil_args, return_stderr=True)
            num_objects = 3
            total_size = len('obj1') + len('.obj2') + len('obj6_')
            CheckUiOutputWithNoMFlag(self, stderr, num_objects, total_size)
            listing1 = TailSet(suri(bucket1_uri), self.FlatListBucket(bucket1_uri))
            listing2 = TailSet(suri(bucket2_uri), self.FlatListBucket(bucket2_uri))
            self.assertEqual(listing1, set(['/obj1', '/.obj2', '/subdir/obj3', '/obj6']))
            self.assertEqual(listing2, set(['/obj1', '/.obj2', '/obj4', '/subdir/obj5', '/obj6']))
            self.assertEqual('.obj2', self.RunGsUtil(['cat', suri(bucket1_uri, '.obj2')], return_stdout=True))
            self.assertEqual('.obj2', self.RunGsUtil(['cat', suri(bucket2_uri, '.obj2')], return_stdout=True))
            self.assertEqual('obj6_', self.RunGsUtil(['cat', suri(bucket2_uri, 'obj6')], return_stdout=True))
        _Check1()

    def test_ui_rsync_bucket_to_bucket_with_m_flag(self):
        """Tests UI output for rsync with -m flag enabled works as expected."""
        self._test_ui_rsync_bucket_to_bucket_helper(gsutil_flags=['-m'])

    def test_ui_rsync_bucket_to_bucket_with_no_m_flag(self):
        """Tests UI output for rsync with -m flag not enabled works as expected."""
        self._test_ui_rsync_bucket_to_bucket_helper()