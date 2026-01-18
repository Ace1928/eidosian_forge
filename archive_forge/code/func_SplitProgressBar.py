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
def SplitProgressBar(original_callback, weights):
    """Splits a progress bar into logical sections.

  Wraps the original callback so that each of the subsections can use the full
  range of 0 to 1 to indicate its progress.  The overall progress bar will
  display total progress based on the weights of the tasks.

  Args:
    original_callback: f(float), The original callback for the progress bar.
    weights: [float], The weights of the tasks to create.  These can be any
      numbers you want and the split will be based on their proportions to
      each other.

  Raises:
    ValueError: If the weights don't add up to 1.

  Returns:
    (f(float), ), A tuple of callback functions, in order, for the subtasks.
  """
    if original_callback is None or original_callback == DefaultProgressBarCallback:
        return tuple([DefaultProgressBarCallback for _ in range(len(weights))])

    def MakeCallback(already_done, weight):

        def Callback(done_fraction):
            original_callback(already_done + done_fraction * weight)
        return Callback
    total = sum(weights)
    callbacks = []
    already_done = 0
    for weight in weights:
        normalized_weight = weight / total
        callbacks.append(MakeCallback(already_done, normalized_weight))
        already_done += normalized_weight
    return tuple(callbacks)