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
def SdkSetup(self):
    """
        Microsoft Windows SDK Setup.

        Return
        ------
        list of str
            paths
        """
    if self.vs_ver > 9.0:
        return []
    return [join(self.si.WindowsSdkDir, 'Setup')]