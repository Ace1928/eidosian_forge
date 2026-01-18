from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import contextlib
import enum
import getpass
import io
import json
import os
import re
import subprocess
import sys
import textwrap
import threading
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_attr
from googlecloudsdk.core.console import console_pager
from googlecloudsdk.core.console import prompt_completer
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import platforms
import six
from six.moves import input  # pylint: disable=redefined-builtin
from six.moves import map  # pylint: disable=redefined-builtin
from six.moves import range  # pylint: disable=redefined-builtin
def ProgressBar(label, stream=log.status, total_ticks=60, first=True, last=True, screen_reader=False):
    """A simple progress bar for tracking completion of an action.

  This progress bar works without having to use any control characters.  It
  prints the action that is being done, and then fills a progress bar below it.
  You should not print anything else on the output stream during this time as it
  will cause the progress bar to break on lines.

  Progress bars can be stacked into a group. first=True marks the first bar in
  the group and last=True marks the last bar in the group. The default assumes
  a singleton bar with first=True and last=True.

  This class can also be used in a context manager.

  Args:
    label: str, The action that is being performed.
    stream: The output stream to write to, stderr by default.
    total_ticks: int, The number of ticks wide to make the progress bar.
    first: bool, True if this is the first bar in a stacked group.
    last: bool, True if this is the last bar in a stacked group.
    screen_reader: bool, override for screen reader accessibility property
      toggle.

  Returns:
    The progress bar.
  """
    style = properties.VALUES.core.interactive_ux_style.Get()
    if style == properties.VALUES.core.InteractiveUXStyles.OFF.name:
        return NoOpProgressBar()
    elif style == properties.VALUES.core.InteractiveUXStyles.TESTING.name:
        return _StubProgressBar(label, stream)
    elif screen_reader or properties.VALUES.accessibility.screen_reader.GetBool():
        return _TextPercentageProgressBar(label, stream)
    else:
        return _NormalProgressBar(label, stream, total_ticks, first, last)