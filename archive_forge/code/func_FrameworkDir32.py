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
def FrameworkDir32(self):
    """
        Microsoft .NET Framework 32bit directory.

        Return
        ------
        str
            path
        """
    guess_fw = join(self.WinDir, 'Microsoft.NET\\Framework')
    return self.ri.lookup(self.ri.vc, 'frameworkdir32') or guess_fw