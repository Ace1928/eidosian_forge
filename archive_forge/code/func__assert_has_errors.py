import os
import textwrap
from unittest import mock
import pycodestyle
from os_win._hacking import checks
from os_win.tests.unit import test_base
def _assert_has_errors(self, code, checker, expected_errors=None, filename=None):
    actual_errors = [e[:3] for e in self._run_check(code, checker, filename)]
    self.assertEqual(expected_errors or [], actual_errors)