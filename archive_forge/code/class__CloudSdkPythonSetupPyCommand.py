from the command line arguments and returns a list of URLs to be given to the
from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import collections
import contextlib
import io
import os
import sys
import textwrap
from googlecloudsdk.api_lib.storage import storage_util
from googlecloudsdk.command_lib.ml_engine import uploads
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import execution_utils
from googlecloudsdk.core import log
from googlecloudsdk.core.util import files
import six
from six.moves import map
from setuptools import setup, find_packages
class _CloudSdkPythonSetupPyCommand(_SetupPyCommand):
    """A command that uses the Cloud SDK Python environment.

  It uses the same OS environment, plus the same PYTHONPATH.

  This is preferred, since it's more controlled.
  """

    def GetArgs(self):
        return execution_utils.ArgsForPythonTool(self.setup_py_path, *self.setup_py_args, python=GetPythonExecutable())

    def GetEnv(self):
        exec_env = os.environ.copy()
        exec_env['PYTHONPATH'] = os.pathsep.join(sys.path)
        return exec_env