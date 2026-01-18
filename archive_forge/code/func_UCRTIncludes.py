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
def UCRTIncludes(self):
    """
        Microsoft Universal C Runtime SDK Include.

        Return
        ------
        list of str
            paths
        """
    if self.vs_ver < 14.0:
        return []
    include = join(self.si.UniversalCRTSdkDir, 'include')
    return [join(include, '%sucrt' % self._ucrt_subdir)]