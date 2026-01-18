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
@property
def OSLibraries(self):
    """
        Microsoft Windows SDK Libraries.

        Return
        ------
        list of str
            paths
        """
    if self.vs_ver <= 10.0:
        arch_subdir = self.pi.target_dir(hidex86=True, x64=True)
        return [join(self.si.WindowsSdkDir, 'Lib%s' % arch_subdir)]
    else:
        arch_subdir = self.pi.target_dir(x64=True)
        lib = join(self.si.WindowsSdkDir, 'lib')
        libver = self._sdk_subdir
        return [join(lib, '%sum%s' % (libver, arch_subdir))]