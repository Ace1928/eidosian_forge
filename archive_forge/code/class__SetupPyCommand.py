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
class _SetupPyCommand(six.with_metaclass(abc.ABCMeta, object)):
    """A command to run setup.py in a given environment.

  Includes the Python version to use and the arguments with which to run
  setup.py.

  Attributes:
    setup_py_path: str, the path to the setup.py file
    setup_py_args: list of str, the arguments with which to call setup.py
    package_root: str, path to the directory containing the package to build
      (must be writable, or setuptools will fail)
  """

    def __init__(self, setup_py_path, setup_py_args, package_root):
        self.setup_py_path = setup_py_path
        self.setup_py_args = setup_py_args
        self.package_root = package_root

    @abc.abstractmethod
    def GetArgs(self):
        """Returns arguments to use for execution (including Python executable)."""
        raise NotImplementedError()

    @abc.abstractmethod
    def GetEnv(self):
        """Returns the environment dictionary to use for Python execution."""
        raise NotImplementedError()

    def Execute(self, out):
        """Run the configured setup.py command.

    Args:
      out: a stream to which the command output should be written.

    Returns:
      int, the return code of the command.
    """
        return execution_utils.Exec(self.GetArgs(), no_exit=True, out_func=out.write, err_func=out.write, cwd=self.package_root, env=self.GetEnv())