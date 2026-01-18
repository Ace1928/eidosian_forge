import datetime
import re
import sys
import threading
import time
import traceback
import unittest
from xml.sax import saxutils
from absl.testing import _pretty_print_reporter
def _print_testcase_details(self, stream):
    for error in self.errors:
        outcome, exception_type, message, error_msg = error
        message = _escape_xml_attr(_safe_str(message))
        exception_type = _escape_xml_attr(str(exception_type))
        error_msg = _escape_cdata(error_msg)
        stream.write('  <%s message="%s" type="%s"><![CDATA[%s]]></%s>\n' % (outcome, message, exception_type, error_msg, outcome))