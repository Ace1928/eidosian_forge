from ctypes.util import find_library
from subprocess import check_output, CalledProcessError, DEVNULL
import ctypes
import os
import sys
import sysconfig
class OpenMP(object):
    """
    Internal representation of the OpenMP module.

    Custom class is used to dynamically add omp runtime function
    to this library when function is called.
    """

    def __init__(self):
        ver = msvc_runtime_major()
        if ver is None:
            self.init_not_msvc()
        else:
            self.init_msvc(ver)

    def init_msvc(self, ver):
        vcomp_path = find_library('vcomp%d.dll' % ver)
        if not vcomp_path:
            raise ImportError("I can't find a shared library for vcomp.")
        else:
            self.libomp = ctypes.CDLL(vcomp_path)
            self.version = 20

    def get_libomp_names(self):
        """Return list of OpenMP libraries to try"""
        return ['omp', 'gomp', 'iomp5']

    def init_not_msvc(self):
        """ Find OpenMP library and try to load if using ctype interface. """
        env_vars = []
        if sys.platform == 'darwin':
            env_vars = ['DYLD_LIBRARY_PATH', 'DYLD_FALLBACK_LIBRARY_PATH']
        else:
            env_vars = ['LD_LIBRARY_PATH']
        paths = []
        for env_var in env_vars:
            env_paths = os.environ.get(env_var, '')
            if env_paths:
                paths.extend(env_paths.split(os.pathsep))
        libomp_names = self.get_libomp_names()
        if cxx is not None:
            for libomp_name in libomp_names:
                cmd = [cxx, '-print-file-name=lib{}{}'.format(libomp_name, get_shared_lib_extension())]
                try:
                    output = check_output(cmd, stderr=DEVNULL)
                    path = os.path.dirname(output.decode().strip())
                    if path:
                        paths.append(path)
                except (OSError, CalledProcessError):
                    pass
        for libomp_name in libomp_names:
            libomp_path = find_library(libomp_name)
            if not libomp_path:
                for path in paths:
                    candidate_path = os.path.join(path, 'lib{}{}'.format(libomp_name, get_shared_lib_extension()))
                    if os.path.isfile(candidate_path):
                        libomp_path = candidate_path
                        break
            if libomp_path:
                try:
                    self.libomp = ctypes.CDLL(libomp_path)
                except OSError:
                    raise ImportError("found openMP library '{}' but couldn't load it. This may happen if you are cross-compiling.".format(libomp_path))
                self.version = 45
                return
        raise ImportError("I can't find a shared library for libomp, you may need to install it or adjust the {} environment variable.".format(env_vars[0]))

    def __getattr__(self, name):
        """
        Get correct function name from libgomp ready to be use.

        __getattr__ is call only `name != libomp` as libomp is a real
        attribute.
        """
        if name == 'VERSION':
            return self.version
        return getattr(self.libomp, 'omp_' + name)