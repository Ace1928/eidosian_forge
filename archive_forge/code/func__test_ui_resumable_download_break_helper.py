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