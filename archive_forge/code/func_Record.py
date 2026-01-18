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
def Record(self, event, flag_names=None, error=None, error_extra_info_json=None):
    """Records the given event.

    Args:
      event: _Event, The event to process.
      flag_names: str, Comma separated list of flag names used with the action.
      error: class, The class (not the instance) of the Exception if a user
        tried to run a command that produced an error.
      error_extra_info_json: {str: json-serializable}, A json serializable dict
        of extra info that we want to log with the error. This enables us to
        write queries that can understand the keys and values in this dict.
    """
    for metrics_reporter in self._metrics_reporters:
        metrics_reporter.Record(event, flag_names=flag_names, error=error, error_extra_info_json=error_extra_info_json)