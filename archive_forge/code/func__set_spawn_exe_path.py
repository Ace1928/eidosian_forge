import multiprocessing
import os
import platform
import sys
import unittest
from absl import app
from absl import logging
from tensorflow.python.eager import test
def _set_spawn_exe_path():
    """Set the path to the executable for spawned processes.

  This utility searches for the binary the parent process is using, and sets
  the executable of multiprocessing's context accordingly.

  Raises:
    RuntimeError: If the binary path cannot be determined.
  """
    if sys.argv[0].endswith('.py'):

        def guess_path(package_root):
            if 'bazel-out' in sys.argv[0] and package_root in sys.argv[0]:
                package_root_base = sys.argv[0][:sys.argv[0].rfind(package_root)]
                binary = os.environ['TEST_TARGET'][2:].replace(':', '/', 1)
                possible_path = os.path.join(package_root_base, package_root, binary)
                logging.info('Guessed test binary path: %s', possible_path)
                if os.access(possible_path, os.X_OK):
                    return possible_path
                return None
        path = guess_path('org_tensorflow')
        if not path:
            path = guess_path('org_keras')
        if path is None:
            logging.error('Cannot determine binary path. sys.argv[0]=%s os.environ=%s', sys.argv[0], os.environ)
            raise RuntimeError('Cannot determine binary path')
        sys.argv[0] = path
    multiprocessing.get_context().set_executable(sys.argv[0])