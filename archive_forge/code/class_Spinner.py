from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import io
import os
import sys
import threading
import time
from googlecloudsdk.calliope import parser_completer
from googlecloudsdk.command_lib.interactive import parser
from googlecloudsdk.command_lib.meta import generate_cli_trees
from googlecloudsdk.core import module_util
from googlecloudsdk.core.console import console_attr
from prompt_toolkit import completion
import six
class Spinner(object):
    """A Spinner to show when completer takes too long to respond.

  Some completer calls take too long, specially those that fetch remote
  resources. An instance of this class can be used as a context manager wrapping
  slow completers to get spinmarks while the completer fetches.

  Attributes:
    _done_loading: Boolean flag indicating whether ticker thread is working.
    _set_spinner: Function reference to InteractiveCliCompleter's spinner
      setter.
    _spin_marks: List of unicode spinmarks to be cycled while loading.
    _ticker: Thread instance that handles displaying the spinner.
    _ticker_index: Integer specifying the last iteration index in _spin_marks.
    _TICKER_INTERVAL: Float specifying time between ticker rotation in
      milliseconds.
    _ticker_length: Integer spcifying length of _spin_marks.
    _TICKER_WAIT: Float specifying the wait time before ticking in milliseconds.
    _TICKER_WAIT_CHECK_INTERVAL: Float specifying interval time to break wait
      in milliseconds.
  """
    _TICKER_INTERVAL = 100
    _TICKER_WAIT = 200
    _TICKER_WAIT_CHECK_INTERVAL = 10

    def __init__(self, set_spinner):
        self._done_loading = False
        self._spin_marks = console_attr.GetConsoleAttr().GetProgressTrackerSymbols().spin_marks
        self._ticker = None
        self._ticker_index = 0
        self._ticker_length = len(self._spin_marks)
        self._set_spinner = set_spinner

    def _Mark(self, spin_mark):
        """Marks spin_mark on stdout and moves cursor back."""
        sys.stdout.write(spin_mark + '\x08')
        sys.stdout.flush()

    def Stop(self):
        """Erases last spin_mark and joins the ticker thread."""
        self._Mark(' ')
        self._done_loading = True
        if self._ticker:
            self._ticker.join()

    def _Ticker(self):
        """Waits for _TICKER_WAIT and then starts printing the spinner."""
        for _ in range(Spinner._TICKER_WAIT // Spinner._TICKER_WAIT_CHECK_INTERVAL):
            time.sleep(Spinner._TICKER_WAIT_CHECK_INTERVAL / 1000.0)
            if self._done_loading:
                break
        while not self._done_loading:
            spin_mark = self._spin_marks[self._ticker_index]
            self._Mark(spin_mark)
            self._ticker_index = (self._ticker_index + 1) % self._ticker_length
            time.sleep(Spinner._TICKER_INTERVAL / 1000.0)

    def __enter__(self):
        self._set_spinner(self)
        self._ticker = threading.Thread(target=self._Ticker)
        self._ticker.start()
        return self

    def __exit__(self, *args):
        self.Stop()
        self._set_spinner(None)