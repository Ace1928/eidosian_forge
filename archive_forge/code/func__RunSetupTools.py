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
def _RunSetupTools(package_root, setup_py_path, output_dir):
    """Executes the setuptools `sdist` command.

  Specifically, runs `python setup.py sdist` (with the full path to `setup.py`
  given by setup_py_path) with arguments to put the final output in output_dir
  and all possible temporary files in a temporary directory. package_root is
  used as the working directory.

  May attempt to run setup.py multiple times with different
  environments/commands if any execution fails:

  1. Using the Cloud SDK Python environment, with a full setuptools invocation
     (`egg_info`, `build`, and `sdist`).
  2. Using the system Python environment, with a full setuptools invocation
     (`egg_info`, `build`, and `sdist`).
  3. Using the Cloud SDK Python environment, with an intermediate setuptools
     invocation (`build` and `sdist`).
  4. Using the system Python environment, with an intermediate setuptools
     invocation (`build` and `sdist`).
  5. Using the Cloud SDK Python environment, with a simple setuptools
     invocation which will also work for plain distutils-based setup.py (just
     `sdist`).
  6. Using the system Python environment, with a simple setuptools
     invocation which will also work for plain distutils-based setup.py (just
     `sdist`).

  The reason for this order is that it prefers first the setup.py invocations
  which leave the fewest files on disk. Then, we prefer the Cloud SDK execution
  environment as it will be the most stable.

  package_root must be writable, or setuptools will fail (there are
  temporary files from setuptools that get put in the CWD).

  Args:
    package_root: str, the directory containing the package (that is, the
      *parent* of the package itself).
    setup_py_path: str, the path to the `setup.py` file to execute.
    output_dir: str, path to a directory in which the built packages should be
      created.

  Returns:
    list of str, the full paths to the generated packages.

  Raises:
    SysExecutableMissingError: if sys.executable is None
    RuntimeError: if the execution of setuptools exited non-zero.
  """
    with _TempDirOrBackup(package_root) as working_dir:
        sdist_args = ['sdist', '--dist-dir', output_dir]
        build_args = ['build', '--build-base', working_dir, '--build-temp', working_dir]
        egg_info_args = ['egg_info', '--egg-base', working_dir]
        setup_py_arg_sets = (egg_info_args + build_args + sdist_args, build_args + sdist_args, sdist_args)
        setup_py_commands = []
        for setup_py_args in setup_py_arg_sets:
            setup_py_commands.append(_CloudSdkPythonSetupPyCommand(setup_py_path, setup_py_args, package_root))
            setup_py_commands.append(_SystemPythonSetupPyCommand(setup_py_path, setup_py_args, package_root))
        for setup_py_command in setup_py_commands:
            out = io.StringIO()
            return_code = setup_py_command.Execute(out)
            if not return_code:
                break
        else:
            raise RuntimeError(out.getvalue())
    local_paths = [os.path.join(output_dir, rel_file) for rel_file in os.listdir(output_dir)]
    log.debug('Python packaging resulted in [%s]', ', '.join(local_paths))
    return local_paths