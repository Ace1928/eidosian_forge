import random
import unittest
import xmlrunner
from __future__ import absolute_import
import os
import sys
import time
from unittest import TestResult, TextTestResult, TextTestRunner
import xml.dom.minidom
def _report_output(test_runner, xml_testsuite, xml_document, stdout, stderr):
    """Appends the system-out and system-err sections to the XML document."""
    systemout = xml_document.createElement('system-out')
    xml_testsuite.appendChild(systemout)
    systemout_text = xml_document.createCDATAOrText(stdout)
    systemout.appendChild(systemout_text)
    systemerr = xml_document.createElement('system-err')
    xml_testsuite.appendChild(systemerr)
    systemerr_text = xml_document.createCDATAOrText(stderr)
    systemerr.appendChild(systemerr_text)