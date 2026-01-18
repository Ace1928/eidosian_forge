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
def MSBuild(self):
    """
        Microsoft Build Engine.

        Return
        ------
        list of str
            paths
        """
    if self.vs_ver < 12.0:
        return []
    elif self.vs_ver < 15.0:
        base_path = self.si.ProgramFilesx86
        arch_subdir = self.pi.current_dir(hidex86=True)
    else:
        base_path = self.si.VSInstallDir
        arch_subdir = ''
    path = 'MSBuild\\%0.1f\\bin%s' % (self.vs_ver, arch_subdir)
    build = [join(base_path, path)]
    if self.vs_ver >= 15.0:
        build += [join(base_path, path, 'Roslyn')]
    return build