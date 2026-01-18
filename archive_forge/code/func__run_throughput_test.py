from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import os
import socket
import sys
import six
import boto
from gslib.commands.perfdiag import _GenerateFileData
import gslib.tests.testcase as testcase
from gslib.tests.testcase.integration_testcase import SkipForXML
from gslib.tests.util import ObjectToURI as suri
from gslib.tests.util import RUN_S3_TESTS
from gslib.tests.util import unittest
from gslib.utils.system_util import IS_WINDOWS
from six import add_move, MovedModule
from six.moves import mock
def _run_throughput_test(self, test_name, num_processes, num_threads, parallelism_strategy=None, compression_ratio=None):
    bucket_uri = self.CreateBucket()
    cmd = ['perfdiag', '-n', str(num_processes * num_threads), '-s', '1024', '-c', str(num_processes), '-k', str(num_threads), '-t', test_name]
    if compression_ratio is not None:
        cmd += ['-j', str(compression_ratio)]
    if parallelism_strategy is not None:
        cmd += ['-p', parallelism_strategy]
    cmd += [suri(bucket_uri)]
    stderr_default = self.RunGsUtil(cmd, return_stderr=True)
    stderr_custom = None
    if self._should_run_with_custom_endpoints():
        stderr_custom = self.RunGsUtil(self._custom_endpoint_flags + cmd, return_stderr=True)
    self.AssertNObjectsInBucket(bucket_uri, 0, versioned=True)
    return (stderr_default, stderr_custom)