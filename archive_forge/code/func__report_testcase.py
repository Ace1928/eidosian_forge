import random
import unittest
import xmlrunner
from __future__ import absolute_import
import os
import sys
import time
from unittest import TestResult, TextTestResult, TextTestRunner
import xml.dom.minidom
def _report_testcase(suite_name, test_result, xml_testsuite, xml_document):
    """Appends a testcase section to the XML document."""
    testcase = xml_document.createElement('testcase')
    xml_testsuite.appendChild(testcase)
    testcase.setAttribute('classname', str(suite_name))
    testcase.setAttribute('name', test_result.test_method.shortDescription() or getattr(test_result.test_method, '_testMethodName', str(test_result.test_method)))
    testcase.setAttribute('time', '%.3f' % test_result.get_elapsed_time())
    if test_result.outcome != _TestInfo.SUCCESS:
        elem_name = ('failure', 'error')[test_result.outcome - 1]
        failure = xml_document.createElement(elem_name)
        testcase.appendChild(failure)
        failure.setAttribute('type', str(test_result.err[0].__name__))
        failure.setAttribute('message', str(test_result.err[1]))
        error_info = test_result.get_error_info()
        failureText = xml_document.createCDATAOrText(error_info)
        failure.appendChild(failureText)