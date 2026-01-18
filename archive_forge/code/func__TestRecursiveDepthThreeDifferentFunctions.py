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
def _TestRecursiveDepthThreeDifferentFunctions(self, process_count, thread_count):
    """Tests recursive application of Apply.

    Calls Apply(A), where A calls Apply(B) followed by Apply(C) and B calls
    Apply(C).

    Args:
      process_count: Number of processes to use.
      thread_count: Number of threads to use.
    """
    base_args = [3, 1, 4, 1, 5]
    args = [[process_count, thread_count, count] for count in base_args]
    results = self._RunApply(_ReApplyWithReplicatedArguments, args, process_count, thread_count)
    self.assertEqual(7 * (sum(base_args) + len(base_args)), sum(results))