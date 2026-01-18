import errno
import os
import select
import socket
import sys
import tempfile
import time
from io import BytesIO
from .. import errors, osutils, tests, trace, win32utils
from . import features, file_utils, test__walkdirs_win32
from .scenarios import load_tests_apply_scenarios
class TestFailedToLoadExtension(tests.TestCase):

    def _try_loading(self):
        try:
            import breezy._fictional_extension_py
        except ImportError as e:
            osutils.failed_to_load_extension(e)
            return True

    def setUp(self):
        super().setUp()
        self.overrideAttr(osutils, '_extension_load_failures', [])

    def test_failure_to_load(self):
        self._try_loading()
        self.assertLength(1, osutils._extension_load_failures)
        self.assertEqual(osutils._extension_load_failures[0], "No module named 'breezy._fictional_extension_py'")

    def test_report_extension_load_failures_no_warning(self):
        self.assertTrue(self._try_loading())
        warnings, result = self.callCatchWarnings(osutils.report_extension_load_failures)
        self.assertLength(0, warnings)

    def test_report_extension_load_failures_message(self):
        log = BytesIO()
        trace.push_log_file(log)
        self.assertTrue(self._try_loading())
        osutils.report_extension_load_failures()
        self.assertContainsRe(log.getvalue(), b'brz: warning: some compiled extensions could not be loaded; see ``brz help missing-extensions``\n')