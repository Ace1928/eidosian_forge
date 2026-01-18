from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import logging
import os
import re
import signal
import sys
import traceback
from gslib import metrics
from gslib.exception import ControlCException
from gslib.utils.constants import UTF8
from gslib.utils.system_util import IS_WINDOWS
def InitializeSignalHandling():
    """Initializes global signal handling.

  Sets up global signal handler for each signal we handle.
  """
    for signal_num in GetCaughtSignals():
        _non_final_signal_handlers[signal_num] = []
        signal.signal(signal_num, _SignalHandler)