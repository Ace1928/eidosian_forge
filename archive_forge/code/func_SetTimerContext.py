from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import contextlib
import json
import os
import pickle
import platform
import socket
import subprocess
import sys
import tempfile
import time
from googlecloudsdk.core import config
from googlecloudsdk.core import execution_utils
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_attr
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import platforms
import six
import six.moves.urllib.error
import six.moves.urllib.parse
import six.moves.urllib.request
def SetTimerContext(self, category, action, label=None, flag_names=None):
    """Sets the context for which the timer is collecting timed events.

    Args:
      category: str, Category of the action being timed.
      action: str, Name of the action being timed.
      label: str, Additional information about the action being timed.
      flag_names: str, Comma separated list of flag names used with the action.
    """
    if category is _COMMANDS_CATEGORY and self._action_level != 0:
        return
    if category is _ERROR_CATEGORY and self._action_level != 0:
        _, action, _, _ = self._timer.GetContext()
    self._timer.SetContext(category, action, label, flag_names)