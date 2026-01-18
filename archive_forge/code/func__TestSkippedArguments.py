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
def _TestSkippedArguments(self, process_count, thread_count):
    n = 2 * process_count * thread_count
    args = range(1, n + 1)
    results = self._RunApply(_ReturnOneValue, args, process_count, thread_count, arg_checker=_SkipEvenNumbersArgChecker)
    self.assertEqual(n / 2, len(results))
    self.assertEqual(n / 2, sum(results))
    args = [2 * x for x in args]
    results = self._RunApply(_ReturnOneValue, args, process_count, thread_count, arg_checker=_SkipEvenNumbersArgChecker)
    self.assertEqual(0, len(results))