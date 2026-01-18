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
def Error(command_path, error, flag_names=None, error_extra_info=None):
    """Logs that a top level Exception was caught for a gcloud command.

  Args:
    command_path: str, The '.' separated name of the calliope command.
    error: class, The class (not the instance) of the exception that was
      caught.
    flag_names: [str], The names of the flags that were used during this
      execution.
    error_extra_info: {str: json-serializable}, A json serializable dict of
      extra info that we want to log with the error. This enables us to write
      queries that can understand the keys and values in this dict.
  """
    _RecordEventAndSetTimerContext(_ERROR_CATEGORY, command_path, _GetExceptionName(error), flag_names=_GetFlagNameString(flag_names), error_extra_info_json=_GetErrorExtraInfo(error_extra_info))