from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import collections
import os
import signal
import sys
import threading
import time
import enum
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_attr
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.console import multiline
from googlecloudsdk.core.console.style import parser
import six
class _NormalProgressTracker(_BaseProgressTracker):
    """A context manager for telling the user about long-running progress."""

    def __enter__(self):
        self._SetupOutput()
        return super(_NormalProgressTracker, self).__enter__()

    def _SetupOutput(self):

        def _FormattedCallback():
            if self._detail_message_callback:
                detail_message = self._detail_message_callback()
                if detail_message:
                    if self._no_spacing:
                        return detail_message
                    return ' ' + detail_message + '...'
            return None
        self._console_output = multiline.SimpleSuffixConsoleOutput(self._stream)
        self._console_message = self._console_output.AddMessage(self._prefix, detail_message_callback=_FormattedCallback)

    def Tick(self):
        """Give a visual indication to the user that some progress has been made.

    Output is sent to sys.stderr. Nothing is shown if output is not a TTY.

    Returns:
      Whether progress has completed.
    """
        with self._lock:
            if not self._done:
                self._ticks += 1
                self._Print(self._GetSuffix())
        self._stream.flush()
        return self._done

    def _GetSuffix(self):
        if self.spinner_override_message:
            num_dots = self._ticks % 4
            return self.spinner_override_message + '.' * num_dots
        else:
            return self._symbols.spin_marks[self._ticks % len(self._symbols.spin_marks)]

    def _Print(self, message=''):
        """Reprints the prefix followed by an optional message.

    If there is a multiline message, we print the full message and every
    time the Prefix Message is the same, we only reprint the last line to
    account for a different 'message'. If there is a new message, we print
    on a new line.

    Args:
      message: str, suffix of message
    """
        if self._spinner_only or not self._output_enabled:
            return
        self._console_output.UpdateMessage(self._console_message, message)
        self._console_output.UpdateConsole()