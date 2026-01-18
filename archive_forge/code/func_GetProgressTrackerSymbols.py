from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import locale
import os
import sys
import unicodedata
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_attr_os
from googlecloudsdk.core.console.style import text
from googlecloudsdk.core.util import encoding as encoding_util
import six
def GetProgressTrackerSymbols(self):
    """Returns the progress tracker characters object.

    Returns:
      A ProgressTrackerSymbols object for the console output device.
    """
    return self._progress_tracker_symbols