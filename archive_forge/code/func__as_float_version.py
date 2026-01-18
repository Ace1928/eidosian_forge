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
@staticmethod
def _as_float_version(version):
    """
        Return a string version as a simplified float version (major.minor)

        Parameters
        ----------
        version: str
            Version.

        Return
        ------
        float
            version
        """
    return float('.'.join(version.split('.')[:2]))