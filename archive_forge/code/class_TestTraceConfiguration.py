import errno
import logging
import os
import re
import sys
import tempfile
from io import StringIO
from .. import debug, errors, trace
from ..trace import (_rollover_trace_maybe, be_quiet, get_verbosity_level,
from . import TestCase, TestCaseInTempDir, TestSkipped, features
class TestTraceConfiguration(TestCaseInTempDir):

    def test_default_config(self):
        config = trace.DefaultConfig()
        self.overrideAttr(trace, '_brz_log_filename', None)
        trace._brz_log_filename = None
        expected_filename = trace._get_brz_log_filename()
        self.assertEqual(None, trace._brz_log_filename)
        config.__enter__()
        try:
            self.assertEqual(expected_filename, trace._brz_log_filename)
        finally:
            config.__exit__(None, None, None)
            self.assertEqual(None, trace._brz_log_filename)