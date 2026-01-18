import errno
import os
import shutil
import stat
import sys
import tempfile
import pyomo.common.envvar as envvar
from pyomo.common.fileutils import this_file_dir, find_executable
class _CMakeBuild(build_ext, object):

    def run(self):
        for cmake_ext in self.extensions:
            self._cmake_build_target(cmake_ext)

    def _cmake_build_target(self, cmake_ext):
        cmake_config = 'Debug' if self.debug else 'Release'
        cmake_args = ['-DCMAKE_INSTALL_PREFIX=' + envvar.PYOMO_CONFIG_DIR] + cmake_ext.user_args
        try:
            sys.stderr.flush()
            sys.stdout.flush()
            old_stderr = os.dup(sys.stderr.fileno())
            os.dup2(sys.stdout.fileno(), sys.stderr.fileno())
            old_environ = dict(os.environ)
            if cmake_ext.parallel:
                os.environ['CMAKE_BUILD_PARALLEL_LEVEL'] = str(cmake_ext.parallel)
            cmake = find_executable('cmake')
            if cmake is None:
                raise IOError('cmake not found in the system PATH')
            self.spawn([cmake, cmake_ext.target_dir] + cmake_args)
            if not self.dry_run:
                self.spawn([cmake, '--build', '.', '--target', 'install', '--config', cmake_config])
        finally:
            sys.stderr.flush()
            sys.stdout.flush()
            os.dup2(old_stderr, sys.stderr.fileno())
            os.environ = old_environ