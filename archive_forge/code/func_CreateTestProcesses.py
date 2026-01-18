from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
from collections import namedtuple
import logging
import os
import subprocess
import re
import sys
import tempfile
import textwrap
import time
import traceback
import six
from six.moves import range
import gslib
from gslib.cloud_api import ProjectIdException
from gslib.command import Command
from gslib.command import ResetFailureCount
from gslib.exception import CommandException
from gslib.project_id import PopulateProjectId
import gslib.tests as tests
from gslib.tests.util import GetTestNames
from gslib.tests.util import InvokedFromParFile
from gslib.tests.util import unittest
from gslib.utils.constants import NO_MAX
from gslib.utils.constants import UTF8
from gslib.utils.system_util import IS_WINDOWS
def CreateTestProcesses(parallel_tests, test_index, process_list, process_done, max_parallel_tests, root_coverage_file=None):
    """Creates test processes to run tests in parallel.

  Args:
    parallel_tests: List of all parallel tests.
    test_index: List index of last created test before this function call.
    process_list: List of running subprocesses. Created processes are appended
                  to this list.
    process_done: List of booleans indicating process completion. One 'False'
                  will be added per process created.
    max_parallel_tests: Maximum number of tests to run in parallel.
    root_coverage_file: The root .coverage filename if coverage is requested.

  Returns:
    Index of last created test.
  """
    orig_test_index = test_index
    executable_prefix = [sys.executable] if not InvokedFromParFile() else []
    s3_argument = ['-s'] if tests.util.RUN_S3_TESTS else []
    multiregional_buckets = ['-b'] if tests.util.USE_MULTIREGIONAL_BUCKETS else []
    project_id_arg = []
    try:
        project_id_arg = ['-o', 'GSUtil:default_project_id=%s' % PopulateProjectId()]
    except ProjectIdException:
        pass
    process_create_start_time = time.time()
    last_log_time = process_create_start_time
    while CountFalseInList(process_done) < max_parallel_tests and test_index < len(parallel_tests):
        env = os.environ.copy()
        if root_coverage_file:
            env['GSUTIL_COVERAGE_OUTPUT_FILE'] = root_coverage_file
        envstr = dict()
        cmd = [six.ensure_str(part) for part in list(executable_prefix + [gslib.GSUTIL_PATH] + project_id_arg + ['test'] + s3_argument + multiregional_buckets + ['--' + _SEQUENTIAL_ISOLATION_FLAG] + [parallel_tests[test_index][len('gslib.tests.test_'):]])]
        for k, v in six.iteritems(env):
            envstr[six.ensure_str(k)] = six.ensure_str(v)
        process_list.append(subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=envstr))
        test_index += 1
        process_done.append(False)
        if time.time() - last_log_time > 5:
            print('Created %d new processes (total %d/%d created)' % (test_index - orig_test_index, len(process_list), len(parallel_tests)))
            last_log_time = time.time()
    if test_index == len(parallel_tests):
        print('Test process creation finished (%d/%d created)' % (len(process_list), len(parallel_tests)))
    return test_index