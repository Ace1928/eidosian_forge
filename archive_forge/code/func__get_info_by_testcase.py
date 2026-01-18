import random
import unittest
import xmlrunner
from __future__ import absolute_import
import os
import sys
import time
from unittest import TestResult, TextTestResult, TextTestRunner
import xml.dom.minidom
def _get_info_by_testcase(self):
    """This method organizes test results by TestCase module. This
        information is used during the report generation, where a XML report
        will be generated for each TestCase.
        """
    tests_by_testcase = {}
    for tests in (self.successes, self.failures, self.errors):
        for test_info in tests:
            if not isinstance(test_info, _TestInfo):
                print('Unexpected test result type: %r' % (test_info,))
                continue
            testcase = type(test_info.test_method)
            module = testcase.__module__ + '.'
            if module == '__main__.':
                module = ''
            testcase_name = module + testcase.__name__
            if testcase_name not in tests_by_testcase:
                tests_by_testcase[testcase_name] = []
            tests_by_testcase[testcase_name].append(test_info)
    return tests_by_testcase