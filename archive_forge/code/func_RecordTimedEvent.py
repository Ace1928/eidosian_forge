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
def RecordTimedEvent(self, name, record_only_on_top_level=False, event_time=None):
    """Records the time when a particular event happened.

    Args:
      name: str, Name of the event.
      record_only_on_top_level: bool, Whether to record only on top level.
      event_time: float, Time when the event happened in secs since epoch.
    """
    if self._action_level == 0 or not record_only_on_top_level:
        self._timer.Event(name, event_time=event_time)