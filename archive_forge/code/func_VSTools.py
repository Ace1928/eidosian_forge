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
def VSTools(self):
    """
        Microsoft Visual Studio Tools.

        Return
        ------
        list of str
            paths
        """
    paths = ['Common7\\IDE', 'Common7\\Tools']
    if self.vs_ver >= 14.0:
        arch_subdir = self.pi.current_dir(hidex86=True, x64=True)
        paths += ['Common7\\IDE\\CommonExtensions\\Microsoft\\TestWindow']
        paths += ['Team Tools\\Performance Tools']
        paths += ['Team Tools\\Performance Tools%s' % arch_subdir]
    return [join(self.si.VSInstallDir, path) for path in paths]