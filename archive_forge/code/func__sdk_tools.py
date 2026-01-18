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
def _sdk_tools(self):
    """
        Microsoft Windows SDK Tools paths generator.

        Return
        ------
        generator of str
            paths
        """
    if self.vs_ver < 15.0:
        bin_dir = 'Bin' if self.vs_ver <= 11.0 else 'Bin\\x86'
        yield join(self.si.WindowsSdkDir, bin_dir)
    if not self.pi.current_is_x86():
        arch_subdir = self.pi.current_dir(x64=True)
        path = 'Bin%s' % arch_subdir
        yield join(self.si.WindowsSdkDir, path)
    if self.vs_ver in (10.0, 11.0):
        if self.pi.target_is_x86():
            arch_subdir = ''
        else:
            arch_subdir = self.pi.current_dir(hidex86=True, x64=True)
        path = 'Bin\\NETFX 4.0 Tools%s' % arch_subdir
        yield join(self.si.WindowsSdkDir, path)
    elif self.vs_ver >= 15.0:
        path = join(self.si.WindowsSdkDir, 'Bin')
        arch_subdir = self.pi.current_dir(x64=True)
        sdkver = self.si.WindowsSdkLastVersion
        yield join(path, '%s%s' % (sdkver, arch_subdir))
    if self.si.WindowsSDKExecutablePath:
        yield self.si.WindowsSDKExecutablePath