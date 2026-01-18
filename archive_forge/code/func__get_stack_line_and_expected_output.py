import io
import os
import tempfile
import unittest
from testtools import TestCase
from testtools.compat import (
from testtools.content import (
from testtools.content_type import (
from testtools.matchers import (
from testtools.tests.helpers import an_exc_info
def _get_stack_line_and_expected_output(self):
    stack_lines = [('/path/to/file', 42, 'some_function', 'print("Hello World")')]
    expected = '  File "/path/to/file", line 42, in some_function\n    print("Hello World")\n'
    return (stack_lines, expected)