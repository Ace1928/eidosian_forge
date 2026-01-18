import argparse
import collections
import functools
import glob
import inspect
import itertools
import os
import re
import subprocess
import sys
import threading
import unicodedata
from enum import (
from typing import (
from . import (
from .argparse_custom import (
def _reader_thread_func(self, read_stdout: bool) -> None:
    """
        Thread function that reads a stream from the process
        :param read_stdout: if True, then this thread deals with stdout. Otherwise it deals with stderr.
        """
    if read_stdout:
        read_stream = self._proc.stdout
        write_stream = self._stdout
    else:
        read_stream = self._proc.stderr
        write_stream = self._stderr
    assert read_stream is not None
    while self._proc.poll() is None:
        available = read_stream.peek()
        if available:
            read_stream.read(len(available))
            self._write_bytes(write_stream, available)