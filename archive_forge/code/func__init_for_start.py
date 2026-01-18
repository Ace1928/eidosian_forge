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
def _init_for_start(self) -> None:
    """Initialization for start()"""
    concurrency: list[str] = self.config.concurrency or []
    if 'multiprocessing' in concurrency:
        if self.config.config_file is None:
            raise ConfigError('multiprocessing requires a configuration file')
        patch_multiprocessing(rcfile=self.config.config_file)
    dycon = self.config.dynamic_context
    if not dycon or dycon == 'none':
        context_switchers = []
    elif dycon == 'test_function':
        context_switchers = [should_start_context_test_function]
    else:
        raise ConfigError(f"Don't understand dynamic_context setting: {dycon!r}")
    context_switchers.extend((plugin.dynamic_context for plugin in self._plugins.context_switchers))
    should_start_context = combine_context_switchers(context_switchers)
    self._collector = Collector(should_trace=self._should_trace, check_include=self._check_include_omit_etc, should_start_context=should_start_context, file_mapper=self._file_mapper, timid=self.config.timid, branch=self.config.branch, warn=self._warn, concurrency=concurrency, metacov=self._metacov)
    suffix = self._data_suffix_specified
    if suffix:
        if not isinstance(suffix, str):
            suffix = True
    elif self.config.parallel:
        if suffix is None:
            suffix = True
        elif not isinstance(suffix, str):
            suffix = bool(suffix)
    else:
        suffix = None
    self._init_data(suffix)
    assert self._data is not None
    self._collector.use_data(self._data, self.config.context)
    if self._plugins.file_tracers and (not self._collector.supports_plugins):
        self._warn("Plugin file tracers ({}) aren't supported with {}".format(', '.join((plugin._coverage_plugin_name for plugin in self._plugins.file_tracers)), self._collector.tracer_name()))
        for plugin in self._plugins.file_tracers:
            plugin._coverage_enabled = False
    self._inorout = InOrOut(config=self.config, warn=self._warn, debug=self._debug if self._debug.should('trace') else None, include_namespace_packages=self.config.include_namespace_packages)
    self._inorout.plugins = self._plugins
    self._inorout.disp_class = self._collector.file_disposition_class
    self._should_write_debug = True
    atexit.register(self._atexit)
    if self.config.sigterm:
        is_main = threading.current_thread() == threading.main_thread()
        if is_main and (not env.WINDOWS):
            self._old_sigterm = signal.signal(signal.SIGTERM, self._on_sigterm)