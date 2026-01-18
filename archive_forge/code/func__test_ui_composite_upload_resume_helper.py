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