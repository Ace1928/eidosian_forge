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
def _get_file_reporter(self, morf: TMorf) -> FileReporter:
    """Get a FileReporter for a module or file name."""
    assert self._data is not None
    plugin = None
    file_reporter: str | FileReporter = 'python'
    if isinstance(morf, str):
        mapped_morf = self._file_mapper(morf)
        plugin_name = self._data.file_tracer(mapped_morf)
        if plugin_name:
            plugin = self._plugins.get(plugin_name)
            if plugin:
                file_reporter = plugin.file_reporter(mapped_morf)
                if file_reporter is None:
                    raise PluginError('Plugin {!r} did not provide a file reporter for {!r}.'.format(plugin._coverage_plugin_name, morf))
    if file_reporter == 'python':
        file_reporter = PythonFileReporter(morf, self)
    assert isinstance(file_reporter, FileReporter)
    return file_reporter