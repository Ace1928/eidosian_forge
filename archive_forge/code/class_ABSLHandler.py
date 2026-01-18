from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import getpass
import io
import itertools
import logging
import os
import socket
import struct
import sys
import time
import timeit
import traceback
import types
import warnings
from absl import flags
from absl._collections_abc import abc
from absl.logging import converter
import six
class ABSLHandler(logging.Handler):
    """Abseil Python logging module's log handler."""

    def __init__(self, python_logging_formatter):
        super(ABSLHandler, self).__init__()
        self._python_handler = PythonHandler(formatter=python_logging_formatter)
        self.activate_python_handler()

    def format(self, record):
        return self._current_handler.format(record)

    def setFormatter(self, fmt):
        self._current_handler.setFormatter(fmt)

    def emit(self, record):
        self._current_handler.emit(record)

    def flush(self):
        self._current_handler.flush()

    def close(self):
        super(ABSLHandler, self).close()
        self._current_handler.close()

    def handle(self, record):
        rv = self.filter(record)
        if rv:
            return self._current_handler.handle(record)
        return rv

    @property
    def python_handler(self):
        return self._python_handler

    def activate_python_handler(self):
        """Uses the Python logging handler as the current logging handler."""
        self._current_handler = self._python_handler

    def use_absl_log_file(self, program_name=None, log_dir=None):
        self._current_handler.use_absl_log_file(program_name, log_dir)

    def start_logging_to_file(self, program_name=None, log_dir=None):
        self._current_handler.start_logging_to_file(program_name, log_dir)