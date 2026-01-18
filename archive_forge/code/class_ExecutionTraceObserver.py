import gzip
import json
import os
import tempfile
from enum import Enum
from functools import partial
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple
from warnings import warn
import torch
import torch.autograd.profiler as prof
from torch._C import _get_privateuse1_backend_name
from torch._C._profiler import (
from torch.autograd import kineto_available, ProfilerActivity
from torch.profiler._memory_profiler import MemoryProfile, MemoryProfileTimeline
class ExecutionTraceObserver:
    """Execution Trace Observer

    Each process can have a single ExecutionTraceObserver instance. The observer
    can be added to record function callbacks via calling register_callback()
    explicitly. Without calling unregister_callback(), repeated calls to
    register_callback() will not add additional observers to record function
    callbacks. Once an ExecutionTraceObserver is created, the start() and stop()
    methods control when the event data is recorded.

    Deleting or calling unregister_callback() will remove the observer from the
    record function callbacks, finalize the output file, and will stop
    incurring any overheads.
    """

    def __init__(self):
        """
        Initializes the default states.
        """
        self._registered = False
        self._execution_trace_running = False

    def __del__(self):
        """
        Calls unregister_callback() to make sure to finalize outputs.
        """
        self.unregister_callback()

    def register_callback(self, output_file_path: str):
        """
        Adds ET observer to record function callbacks. The data will be
        written to output_file_path.
        """
        if not self._registered:
            self._output_file_path = output_file_path
            self._registered = _add_execution_trace_observer(output_file_path)

    def unregister_callback(self):
        """
        Removes ET observer from record function callbacks.
        """
        if self._registered:
            self.stop()
            _remove_execution_trace_observer()
            self._registered = False

    @property
    def is_registered(self):
        """
        Returns True if the execution trace observer is registered, otherwise False.
        """
        return self._registered

    def is_running(self):
        """
        Returns True if the observer is running, otherwise False.
        """
        return self._execution_trace_running

    def start(self):
        """
        Starts to capture.
        """
        if self._registered and (not self._execution_trace_running):
            _enable_execution_trace_observer()
            self._execution_trace_running = True

    def stop(self):
        """
        Stops to capture.
        """
        if self._execution_trace_running:
            _disable_execution_trace_observer()
            self._execution_trace_running = False

    def get_output_file_path(self) -> str:
        """
        Returns the output file name.
        """
        if self.is_registered:
            return self._output_file_path
        else:
            raise RuntimeError('A callback to the ET profiler needs to be registered first before getting the output file path')