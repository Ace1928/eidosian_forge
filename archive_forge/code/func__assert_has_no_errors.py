import os
import textwrap
from unittest import mock
import pycodestyle
from os_win._hacking import checks
from os_win.tests.unit import test_base
def _assert_has_no_errors(self, code, checker, filename=None):
    self._assert_has_errors(code, checker, filename=filename)