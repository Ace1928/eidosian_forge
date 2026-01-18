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
def OSIncludes(self):
    """
        Microsoft Windows SDK Include.

        Return
        ------
        list of str
            paths
        """
    include = join(self.si.WindowsSdkDir, 'include')
    if self.vs_ver <= 10.0:
        return [include, join(include, 'gl')]
    else:
        if self.vs_ver >= 14.0:
            sdkver = self._sdk_subdir
        else:
            sdkver = ''
        return [join(include, '%sshared' % sdkver), join(include, '%sum' % sdkver), join(include, '%swinrt' % sdkver)]