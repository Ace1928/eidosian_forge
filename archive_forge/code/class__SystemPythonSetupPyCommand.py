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
class _SystemPythonSetupPyCommand(_SetupPyCommand):
    """A command that uses the system Python environment.

  Uses the same executable as the Cloud SDK.

  Important in case of e.g. a setup.py file that has non-stdlib dependencies.
  """

    def GetArgs(self):
        return [GetPythonExecutable(), self.setup_py_path] + self.setup_py_args

    def GetEnv(self):
        return None