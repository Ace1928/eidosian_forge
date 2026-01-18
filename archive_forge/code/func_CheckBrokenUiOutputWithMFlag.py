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
def CheckBrokenUiOutputWithMFlag(test_case, content, num_objects, total_size=0, metadata=False):
    """Checks if the UI output fails as expected with the -m flag enabled.

  Args:
    test_case: Testcase used to maintain the same assert structure.
    content: The output provided by the UI.
    num_objects: The number of objects processed.
    total_size: The total size transferred in the operation. Used for data
                operations only.
    metadata: Indicates whether this is a metadata operation.
  """
    description_string = _FindAppropriateDescriptionString(metadata)
    test_case.assertNotIn('100% Done', content)
    files_completed_string = str(num_objects) + '/' + str(num_objects)
    test_case.assertNotIn(files_completed_string + description_string, content)
    if not metadata:
        total_size_string = BytesToFixedWidthString(total_size)
        zero = BytesToFixedWidthString(0)
        test_case.assertIn(zero + '/' + total_size_string, content)
        test_case.assertNotIn(total_size_string + '/' + total_size_string, content)
    final_message_prefix = 'Operation completed over'
    test_case.assertNotIn(final_message_prefix, content)