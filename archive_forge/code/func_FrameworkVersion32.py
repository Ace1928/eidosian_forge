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
def FrameworkVersion32(self):
    """
        Microsoft .NET Framework 32bit versions.

        Return
        ------
        tuple of str
            versions
        """
    return self._find_dot_net_versions(32)