from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import contextlib
import errno
import os
import re
import signal
import subprocess
import sys
import threading
import time
from googlecloudsdk.core import argv_utils
from googlecloudsdk.core import config
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.configurations import named_configs
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import parallel
from googlecloudsdk.core.util import platforms
import six
from six.moves import map
def ArgsForPythonTool(executable_path, *args, **kwargs):
    """Constructs an argument list for calling the Python interpreter.

  Args:
    executable_path: str, The full path to the Python main file.
    *args: args for the command
    **kwargs: python: str, path to Python executable to use (defaults to
      automatically detected)

  Returns:
    An argument list to execute the Python interpreter

  Raises:
    TypeError: if an unexpected keyword argument is passed
  """
    unexpected_arguments = set(kwargs) - set(['python'])
    if unexpected_arguments:
        raise TypeError("ArgsForPythonTool() got unexpected keyword arguments '[{0}]'".format(', '.join(unexpected_arguments)))
    python_executable = kwargs.get('python') or GetPythonExecutable()
    python_args_str = encoding.GetEncodedValue(os.environ, 'CLOUDSDK_PYTHON_ARGS', '')
    python_args = python_args_str.split()
    return _GetToolArgs(python_executable, python_args, executable_path, *args)