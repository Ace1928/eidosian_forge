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