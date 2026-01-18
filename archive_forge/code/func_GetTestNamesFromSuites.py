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
def GetTestNamesFromSuites(test_suite):
    """Takes a list of test suites and returns a list of contained test names."""
    suites = [test_suite]
    test_names = []
    while suites:
        suite = suites.pop()
        for test in suite:
            if isinstance(test, unittest.TestSuite):
                suites.append(test)
            else:
                test_names.append(test.id()[len('gslib.tests.test_'):])
    return test_names