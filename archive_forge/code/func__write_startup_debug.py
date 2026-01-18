from __future__ import annotations
import atexit
import collections
import contextlib
import os
import os.path
import platform
import signal
import sys
import threading
import time
import warnings
from types import FrameType
from typing import (
from coverage import env
from coverage.annotate import AnnotateReporter
from coverage.collector import Collector, HAS_CTRACER
from coverage.config import CoverageConfig, read_coverage_config
from coverage.context import should_start_context_test_function, combine_context_switchers
from coverage.data import CoverageData, combine_parallel_data
from coverage.debug import (
from coverage.disposition import disposition_debug_msg
from coverage.exceptions import ConfigError, CoverageException, CoverageWarning, PluginError
from coverage.files import PathAliases, abs_file, relative_filename, set_relative_directory
from coverage.html import HtmlReporter
from coverage.inorout import InOrOut
from coverage.jsonreport import JsonReporter
from coverage.lcovreport import LcovReporter
from coverage.misc import bool_or_none, join_regex
from coverage.misc import DefaultValue, ensure_dir_for_file, isolate_module
from coverage.multiproc import patch_multiprocessing
from coverage.plugin import FileReporter
from coverage.plugin_support import Plugins
from coverage.python import PythonFileReporter
from coverage.report import SummaryReporter
from coverage.report_core import render_report
from coverage.results import Analysis
from coverage.types import (
from coverage.xmlreport import XmlReporter
def _write_startup_debug(self) -> None:
    """Write out debug info at startup if needed."""
    wrote_any = False
    with self._debug.without_callers():
        if self._debug.should('config'):
            config_info = self.config.debug_info()
            write_formatted_info(self._debug.write, 'config', config_info)
            wrote_any = True
        if self._debug.should('sys'):
            write_formatted_info(self._debug.write, 'sys', self.sys_info())
            for plugin in self._plugins:
                header = 'sys: ' + plugin._coverage_plugin_name
                info = plugin.sys_info()
                write_formatted_info(self._debug.write, header, info)
            wrote_any = True
        if self._debug.should('pybehave'):
            write_formatted_info(self._debug.write, 'pybehave', env.debug_info())
            wrote_any = True
    if wrote_any:
        write_formatted_info(self._debug.write, 'end', ())