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
class _NormalCompletionProgressTracker(object):
    """A context manager for visual feedback during long-running completions.

  A completion that exceeds the timeout is assumed to be refreshing the cache.
  At that point the progress tracker displays '?', forks the cache operation
  into the background, and exits.  This gives the background cache update a
  chance finish.  After background_ttl more seconds the update is forcibly
  exited (forced to call exit rather than killed by signal) to prevent not
  responding updates from proliferating in the background.
  """
    _COMPLETION_FD = 9

    def __init__(self, ofile, timeout, tick_delay, background_ttl, autotick):
        self._ofile = ofile or self._GetStream()
        self._timeout = timeout
        self._tick_delay = tick_delay
        self.__autotick = autotick
        self._background_ttl = background_ttl
        self._ticks = 0
        self._symbols = console_attr.GetConsoleAttr().GetProgressTrackerSymbols()

    def __enter__(self):
        if self._autotick:
            self._old_handler = signal.signal(signal.SIGALRM, self._Spin)
            self._old_itimer = signal.setitimer(signal.ITIMER_REAL, self._tick_delay, self._tick_delay)
        return self

    def __exit__(self, unused_type=None, unused_value=True, unused_traceback=None):
        if self._autotick:
            signal.setitimer(signal.ITIMER_REAL, *self._old_itimer)
            signal.signal(signal.SIGALRM, self._old_handler)
        if not self._TimedOut():
            self._WriteMark(' ')

    def _TimedOut(self):
        """True if the tracker has timed out."""
        return self._timeout < 0

    def _Spin(self, unused_sig=None, unused_frame=None):
        """Rotates the spinner one tick and checks for timeout."""
        self._ticks += 1
        self._WriteMark(self._symbols.spin_marks[self._ticks % len(self._symbols.spin_marks)])
        self._timeout -= self._tick_delay
        if not self._TimedOut():
            return
        self._WriteMark('?')
        if os.fork():
            os._exit(1)
        signal.signal(signal.SIGALRM, self._ExitBackground)
        signal.setitimer(signal.ITIMER_REAL, self._background_ttl, self._background_ttl)
        self._ofile = None

    def _WriteMark(self, mark):
        """Writes one mark to self._ofile."""
        if self._ofile:
            self._ofile.write(mark + '\x08')
            self._ofile.flush()

    @staticmethod
    def _ExitBackground():
        """Unconditionally exits the background completer process after timeout."""
        os._exit(1)

    @property
    def _autotick(self):
        return self.__autotick

    @staticmethod
    def _GetStream():
        """Returns the completer output stream."""
        return os.fdopen(os.dup(_NormalCompletionProgressTracker._COMPLETION_FD), 'w')