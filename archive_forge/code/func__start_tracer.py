from __future__ import annotations
import functools
import os
import sys
from types import FrameType
from typing import (
from coverage import env
from coverage.config import CoverageConfig
from coverage.data import CoverageData
from coverage.debug import short_stack
from coverage.disposition import FileDisposition
from coverage.exceptions import ConfigError
from coverage.misc import human_sorted_items, isolate_module
from coverage.plugin import CoveragePlugin
from coverage.pytracer import PyTracer
from coverage.sysmon import SysMonitor
from coverage.types import (
def _start_tracer(self) -> TTraceFn | None:
    """Start a new Tracer object, and store it in self.tracers."""
    tracer = self._trace_class(**self._core_kwargs)
    tracer.data = self.data
    tracer.trace_arcs = self.branch
    tracer.should_trace = self.should_trace
    tracer.should_trace_cache = self.should_trace_cache
    tracer.warn = self.warn
    if hasattr(tracer, 'concur_id_func'):
        tracer.concur_id_func = self.concur_id_func
    if hasattr(tracer, 'file_tracers'):
        tracer.file_tracers = self.file_tracers
    if hasattr(tracer, 'threading'):
        tracer.threading = self.threading
    if hasattr(tracer, 'check_include'):
        tracer.check_include = self.check_include
    if hasattr(tracer, 'should_start_context'):
        tracer.should_start_context = self.should_start_context
    if hasattr(tracer, 'switch_context'):
        tracer.switch_context = self.switch_context
    if hasattr(tracer, 'disable_plugin'):
        tracer.disable_plugin = self.disable_plugin
    fn = tracer.start()
    self.tracers.append(tracer)
    return fn