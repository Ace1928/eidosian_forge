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
@CaptureAndLogException
def Executions(command_name, version_string='unknown'):
    """Logs that a top level SDK script was run.

  Args:
    command_name: str, The script name.
    version_string: str, The version of the command.
  """
    _RecordEventAndSetTimerContext(_EXECUTIONS_CATEGORY, command_name, version_string)