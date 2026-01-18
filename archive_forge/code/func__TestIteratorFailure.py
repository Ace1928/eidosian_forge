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
@Timeout
def _TestIteratorFailure(self, process_count, thread_count):
    """Tests apply with a failing iterator."""
    args = FailingIterator(10, [0])
    results = self._RunApply(_ReturnOneValue, args, process_count, thread_count)
    self.assertEqual(9, len(results))
    args = FailingIterator(10, [5])
    results = self._RunApply(_ReturnOneValue, args, process_count, thread_count)
    self.assertEqual(9, len(results))
    args = FailingIterator(10, [9])
    results = self._RunApply(_ReturnOneValue, args, process_count, thread_count)
    self.assertEqual(9, len(results))
    if process_count * thread_count > 1:
        args = FailingIterator(10, [9])
        results = self._RunApply(_ReturnOneValue, args, process_count, thread_count, fail_on_error=True)
        self.assertEqual(9, len(results))
    args = FailingIterator(10, range(10))
    results = self._RunApply(_ReturnOneValue, args, process_count, thread_count)
    self.assertEqual(0, len(results))
    args = FailingIterator(0, [])
    results = self._RunApply(_ReturnOneValue, args, process_count, thread_count)
    self.assertEqual(0, len(results))