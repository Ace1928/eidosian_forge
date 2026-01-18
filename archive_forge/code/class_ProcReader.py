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
class ProcReader:
    """
    Used to capture stdout and stderr from a Popen process if any of those were set to subprocess.PIPE.
    If neither are pipes, then the process will run normally and no output will be captured.
    """

    def __init__(self, proc: PopenTextIO, stdout: Union[StdSim, TextIO], stderr: Union[StdSim, TextIO]) -> None:
        """
        ProcReader initializer
        :param proc: the Popen process being read from
        :param stdout: the stream to write captured stdout
        :param stderr: the stream to write captured stderr
        """
        self._proc = proc
        self._stdout = stdout
        self._stderr = stderr
        self._out_thread = threading.Thread(name='out_thread', target=self._reader_thread_func, kwargs={'read_stdout': True})
        self._err_thread = threading.Thread(name='err_thread', target=self._reader_thread_func, kwargs={'read_stdout': False})
        if self._proc.stdout is not None:
            self._out_thread.start()
        if self._proc.stderr is not None:
            self._err_thread.start()

    def send_sigint(self) -> None:
        """Send a SIGINT to the process similar to if <Ctrl>+C were pressed"""
        import signal
        if sys.platform.startswith('win'):
            self._proc.send_signal(signal.CTRL_BREAK_EVENT)
        else:
            try:
                group_id = os.getpgid(self._proc.pid)
                os.killpg(group_id, signal.SIGINT)
            except ProcessLookupError:
                return

    def terminate(self) -> None:
        """Terminate the process"""
        self._proc.terminate()

    def wait(self) -> None:
        """Wait for the process to finish"""
        if self._out_thread.is_alive():
            self._out_thread.join()
        if self._err_thread.is_alive():
            self._err_thread.join()
        out, err = self._proc.communicate()
        if out:
            self._write_bytes(self._stdout, out)
        if err:
            self._write_bytes(self._stderr, err)

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

    @staticmethod
    def _write_bytes(stream: Union[StdSim, TextIO], to_write: Union[bytes, str]) -> None:
        """
        Write bytes to a stream
        :param stream: the stream being written to
        :param to_write: the bytes being written
        """
        if isinstance(to_write, str):
            to_write = to_write.encode()
        try:
            stream.buffer.write(to_write)
        except BrokenPipeError:
            pass