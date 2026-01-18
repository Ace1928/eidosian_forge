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
def NetFxSDKIncludes(self):
    """
        Microsoft .Net Framework SDK Includes.

        Return
        ------
        list of str
            paths
        """
    if self.vs_ver < 14.0 or not self.si.NetFxSdkDir:
        return []
    return [join(self.si.NetFxSdkDir, 'include\\um')]