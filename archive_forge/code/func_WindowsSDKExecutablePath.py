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
def WindowsSDKExecutablePath(self):
    """
        Microsoft Windows SDK executable directory.

        Return
        ------
        str
            path
        """
    if self.vs_ver <= 11.0:
        netfxver = 35
        arch = ''
    else:
        netfxver = 40
        hidex86 = True if self.vs_ver <= 12.0 else False
        arch = self.pi.current_dir(x64=True, hidex86=hidex86)
    fx = 'WinSDK-NetFx%dTools%s' % (netfxver, arch.replace('\\', '-'))
    regpaths = []
    if self.vs_ver >= 14.0:
        for ver in self.NetFxSdkVersion:
            regpaths += [join(self.ri.netfx_sdk, ver, fx)]
    for ver in self.WindowsSdkVersion:
        regpaths += [join(self.ri.windows_sdk, 'v%sA' % ver, fx)]
    for path in regpaths:
        execpath = self.ri.lookup(path, 'installationfolder')
        if execpath:
            return execpath
    return None