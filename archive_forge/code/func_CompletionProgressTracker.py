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
def CompletionProgressTracker(ofile=None, timeout=4.0, tick_delay=0.1, background_ttl=60.0, autotick=True):
    """A context manager for visual feedback during long-running completions.

  A completion that exceeds the timeout is assumed to be refreshing the cache.
  At that point the progress tracker displays '?', forks the cache operation
  into the background, and exits.  This gives the background cache update a
  chance finish.  After background_ttl more seconds the update is forcibly
  exited (forced to call exit rather than killed by signal) to prevent not
  responding updates from proliferating in the background.

  Args:
    ofile: The stream to write to.
    timeout: float, The amount of time in second to show the tracker before
      backgrounding it.
    tick_delay: float, The time in second between ticks of the spinner.
    background_ttl: float, The number of seconds to allow the completion to
      run in the background before killing it.
    autotick: bool, True to tick the spinner automatically.

  Returns:
    The completion progress tracker.
  """
    style = properties.VALUES.core.interactive_ux_style.Get()
    if style == properties.VALUES.core.InteractiveUXStyles.OFF.name or style == properties.VALUES.core.InteractiveUXStyles.TESTING.name:
        return _NoOpCompletionProgressTracker()
    else:
        return _NormalCompletionProgressTracker(ofile, timeout, tick_delay, background_ttl, autotick)