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
def BuildPackages(package_path, output_dir):
    """Builds Python packages from the given package source.

  That is, builds Python packages from the code in package_path, using its
  parent directory (the 'package root') as its context using the setuptools
  `sdist` command.

  If there is a `setup.py` file in the package root, use that. Otherwise,
  use a simple, temporary one made for this package.

  We try to be as unobstrustive as possible (see _RunSetupTools for details):

  - setuptools writes some files to the package root--we move as many temporary
    generated files out of the package root as possible
  - the final output gets written to output_dir
  - any temporary setup.py file is written outside of the package root.
  - if the current directory isn't writable, we silenly make a temporary copy

  Args:
    package_path: str. Path to the package. This should be the path to
      the directory containing the Python code to be built, *not* its parent
      (which optionally contains setup.py and other metadata).
    output_dir: str, path to a long-lived directory in which the built packages
      should be created.

  Returns:
    list of str. The full local path to all built Python packages.

  Raises:
    SetuptoolsFailedError: If the setup.py file fails to successfully build.
    MissingInitError: If the package doesn't contain an `__init__.py` file.
    InvalidSourceDirError: if the source directory is not valid.
  """
    package_path = os.path.abspath(package_path)
    package_root = os.path.dirname(package_path)
    with _TempDirOrBackup(package_path) as working_dir:
        package_root = _CopyIfNotWritable(package_root, working_dir)
        if not os.path.exists(os.path.join(package_path, '__init__.py')):
            raise MissingInitError(package_path)
        setup_py_path = os.path.join(package_root, 'setup.py')
        package_name = os.path.basename(package_path)
        generated = _GenerateSetupPyIfNeeded(setup_py_path, package_name)
        try:
            return _RunSetupTools(package_root, setup_py_path, output_dir)
        except RuntimeError as err:
            raise SetuptoolsFailedError(six.text_type(err), generated)
        finally:
            if generated:
                pyc_file = os.path.join(package_root, 'setup.pyc')
                for path in (setup_py_path, pyc_file):
                    try:
                        os.unlink(path)
                    except OSError:
                        log.debug("Couldn't remove file [%s] (it may never have been created).", pyc_file)