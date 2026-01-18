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