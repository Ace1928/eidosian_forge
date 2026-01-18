import random
import unittest
import xmlrunner
from __future__ import absolute_import
import os
import sys
import time
from unittest import TestResult, TextTestResult, TextTestRunner
import xml.dom.minidom
class _TestInfo(object):
    """This class is used to keep useful information about the execution of a
    test method.
    """
    SUCCESS, FAILURE, ERROR = range(3)

    def __init__(self, test_result, test_method, outcome=SUCCESS, err=None):
        """Create a new instance of _TestInfo."""
        self.test_result = test_result
        self.test_method = test_method
        self.outcome = outcome
        self.err = err
        self.stdout = test_result.stdout and test_result.stdout.getvalue().strip() or ''
        self.stderr = test_result.stdout and test_result.stderr.getvalue().strip() or ''

    def get_elapsed_time(self):
        """Return the time that shows how long the test method took to
        execute.
        """
        return self.test_result.stop_time - self.test_result.start_time

    def get_description(self):
        """Return a text representation of the test method."""
        return self.test_result.getDescription(self.test_method)

    def get_error_info(self):
        """Return a text representation of an exception thrown by a test
        method.
        """
        if not self.err:
            return ''
        return self.test_result._exc_info_to_string(self.err, self.test_method)