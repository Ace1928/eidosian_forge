import json
from os import listdir, pathsep
from os.path import join, isfile, isdir, dirname
from subprocess import CalledProcessError
import contextlib
import platform
import itertools
import subprocess
import distutils.errors
from setuptools.extern.more_itertools import unique_everseen
def _guess_vc(self):
    """
        Locate Visual C++ for VS2017+.

        Return
        ------
        str
            path
        """
    if self.vs_ver <= 14.0:
        return ''
    try:
        vs_dir = self.known_vs_paths[self.vs_ver]
    except KeyError:
        vs_dir = self.VSInstallDir
    guess_vc = join(vs_dir, 'VC\\Tools\\MSVC')
    try:
        vc_ver = listdir(guess_vc)[-1]
        self.vc_ver = self._as_float_version(vc_ver)
        return join(guess_vc, vc_ver)
    except (OSError, IndexError):
        return ''