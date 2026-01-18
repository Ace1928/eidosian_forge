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
def SplitParallelizableTestSuite(test_suite):
    """Splits a test suite into groups with different running properties.

  Args:
    test_suite: A python unittest test suite.

  Returns:
    4-part tuple of lists of test names:
    (tests that must be run sequentially,
     tests that must be isolated in a separate process but can be run either
         sequentially or in parallel,
     unit tests that can be run in parallel,
     integration tests that can run in parallel)
  """
    from gslib.tests.testcase.unit_testcase import GsUtilUnitTestCase
    isolated_tests = []
    sequential_tests = []
    parallelizable_integration_tests = []
    parallelizable_unit_tests = []
    items_to_evaluate = [test_suite]
    cases_to_evaluate = []
    while items_to_evaluate:
        suite_or_case = items_to_evaluate.pop()
        if isinstance(suite_or_case, unittest.suite.TestSuite):
            for item in suite_or_case._tests:
                items_to_evaluate.append(item)
        elif isinstance(suite_or_case, unittest.TestCase):
            cases_to_evaluate.append(suite_or_case)
    for test_case in cases_to_evaluate:
        test_method = getattr(test_case, test_case._testMethodName, None)
        if getattr(test_method, 'requires_isolation', False):
            isolated_tests.append(TestCaseToName(test_case))
        elif not getattr(test_method, 'is_parallelizable', True):
            sequential_tests.append(TestCaseToName(test_case))
        elif not getattr(test_case, 'is_parallelizable', True):
            sequential_tests.append(TestCaseToName(test_case))
        elif isinstance(test_case, GsUtilUnitTestCase):
            parallelizable_unit_tests.append(TestCaseToName(test_case))
        else:
            parallelizable_integration_tests.append(TestCaseToName(test_case))
    return (sorted(sequential_tests), sorted(isolated_tests), sorted(parallelizable_unit_tests), sorted(parallelizable_integration_tests))