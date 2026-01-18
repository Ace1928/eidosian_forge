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
def _get_file_reporters(self, morfs: Iterable[TMorf] | None=None) -> list[FileReporter]:
    """Get a list of FileReporters for a list of modules or file names.

        For each module or file name in `morfs`, find a FileReporter.  Return
        the list of FileReporters.

        If `morfs` is a single module or file name, this returns a list of one
        FileReporter.  If `morfs` is empty or None, then the list of all files
        measured is used to find the FileReporters.

        """
    assert self._data is not None
    if not morfs:
        morfs = self._data.measured_files()
    if not isinstance(morfs, (list, tuple, set)):
        morfs = [morfs]
    file_reporters = [self._get_file_reporter(morf) for morf in morfs]
    return file_reporters