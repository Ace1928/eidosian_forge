import datetime
import re
import sys
import threading
import time
import traceback
import unittest
from xml.sax import saxutils
from absl.testing import _pretty_print_reporter
class TextAndXMLTestRunner(unittest.TextTestRunner):
    """A test runner that produces both formatted text results and XML.

  It prints out the names of tests as they are run, errors as they
  occur, and a summary of the results at the end of the test run.
  """
    _TEST_RESULT_CLASS = _TextAndXMLTestResult
    _xml_stream = None
    _testsuites_properties = {}

    def __init__(self, xml_stream=None, *args, **kwargs):
        """Initialize a TextAndXMLTestRunner.

    Args:
      xml_stream: file-like or None; XML-formatted test results are output
          via this object's write() method.  If None (the default), the
          new instance behaves as described in the set_default_xml_stream method
          documentation below.
      *args: passed unmodified to unittest.TextTestRunner.__init__.
      **kwargs: passed unmodified to unittest.TextTestRunner.__init__.
    """
        super(TextAndXMLTestRunner, self).__init__(*args, **kwargs)
        if xml_stream is not None:
            self._xml_stream = xml_stream

    @classmethod
    def set_default_xml_stream(cls, xml_stream):
        """Sets the default XML stream for the class.

    Args:
      xml_stream: file-like or None; used for instances when xml_stream is None
          or not passed to their constructors.  If None is passed, instances
          created with xml_stream=None will act as ordinary TextTestRunner
          instances; this is the default state before any calls to this method
          have been made.
    """
        cls._xml_stream = xml_stream

    def _makeResult(self):
        if self._xml_stream is None:
            return super(TextAndXMLTestRunner, self)._makeResult()
        else:
            return self._TEST_RESULT_CLASS(self._xml_stream, self.stream, self.descriptions, self.verbosity, testsuites_properties=self._testsuites_properties)

    @classmethod
    def set_testsuites_property(cls, key, value):
        cls._testsuites_properties[key] = value