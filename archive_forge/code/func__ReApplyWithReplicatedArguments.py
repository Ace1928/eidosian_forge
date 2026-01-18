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
def _ReApplyWithReplicatedArguments(cls, args, thread_state=None):
    """Calls Apply with arguments repeated seven times.

  The first two elements of args should be the process and thread counts,
  respectively, to be used for the recursive calls.

  Args:
    cls: The Command class to call Apply on.
    args: Arguments to pass to Apply.
    thread_state: Unused, required by function signature.

  Returns:
    Number of values returned by the two calls to Apply.
  """
    new_args = [args] * 7
    process_count = _AdjustProcessCountIfWindows(args[0])
    thread_count = args[1]
    return_values = cls.Apply(_PerformNRecursiveCalls, new_args, _ExceptionHandler, arg_checker=DummyArgChecker, process_count=process_count, thread_count=thread_count, should_return_results=True)
    ret = sum(return_values)
    return_values = cls.Apply(_ReturnOneValue, new_args, _ExceptionHandler, arg_checker=DummyArgChecker, process_count=process_count, thread_count=thread_count, should_return_results=True)
    return len(return_values) + ret