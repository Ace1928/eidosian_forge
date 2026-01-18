from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import functools
import os
import signal
import six
import threading
import textwrap
import time
from unittest import mock
import boto
from boto.storage_uri import BucketStorageUri
from boto.storage_uri import StorageUri
from gslib import cs_api_map
from gslib import command
from gslib.command import Command
from gslib.command import CreateOrGetGsutilLogger
from gslib.command import DummyArgChecker
from gslib.tests.mock_cloud_api import MockCloudApi
from gslib.tests.mock_logging_handler import MockLoggingHandler
import gslib.tests.testcase as testcase
from gslib.tests.testcase.base import RequiresIsolation
from gslib.tests.util import unittest
from gslib.utils.parallelism_framework_util import CheckMultiprocessingAvailableAndInit
from gslib.utils.parallelism_framework_util import multiprocessing_context
from gslib.utils.system_util import IS_OSX
from gslib.utils.system_util import IS_WINDOWS
@RequiresIsolation
def _TestApplySaturatesAvailableProcessesAndThreads(self, process_count, thread_count):
    """Tests that created processes and threads evenly share tasks."""
    calls_per_thread = 2
    args = [()] * (process_count * thread_count * calls_per_thread)
    expected_calls_per_thread = calls_per_thread
    if not self.command_class(True).multiprocessing_is_available:
        expected_calls_per_thread = calls_per_thread * process_count
    results = self._RunApply(_SleepThenReturnProcAndThreadId, args, process_count, thread_count)
    usage_dict = {}
    for process_id, thread_id in results:
        usage_dict[process_id, thread_id] = usage_dict.get((process_id, thread_id), 0) + 1
    for id_tuple, num_tasks_completed in six.iteritems(usage_dict):
        self.assertEqual(num_tasks_completed, expected_calls_per_thread, 'Process %s thread %s completed %s tasks. Expected: %s' % (id_tuple[0], id_tuple[1], num_tasks_completed, expected_calls_per_thread))