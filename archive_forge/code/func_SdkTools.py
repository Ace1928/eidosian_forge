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
def SdkTools(self):
    """
        Microsoft Windows SDK Tools.

        Return
        ------
        list of str
            paths
        """
    return list(self._sdk_tools())