import os
import textwrap
from unittest import mock
import pycodestyle
from os_win._hacking import checks
from os_win.tests.unit import test_base
def _run_check(self, code, checker, filename=None):
    mock_checks = {'physical_line': {}, 'logical_line': {}, 'tree': {}}
    with mock.patch('pycodestyle._checks', mock_checks):
        pycodestyle.register_check(checker)
        lines = textwrap.dedent(code).strip().splitlines(True)
        checker = pycodestyle.Checker(filename=filename, lines=lines)
        with mock.patch('pycodestyle.StandardReport.get_file_results'):
            checker.check_all()
        checker.report._deferred_print.sort()
        return checker.report._deferred_print