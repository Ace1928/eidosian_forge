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
def _sdk_subdir(self):
    """
        Microsoft Windows SDK version subdir.

        Return
        ------
        str
            subdir
        """
    ucrtver = self.si.WindowsSdkLastVersion
    return '%s\\' % ucrtver if ucrtver else ''