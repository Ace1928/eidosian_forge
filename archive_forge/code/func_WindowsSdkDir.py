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
def WindowsSdkDir(self):
    """
        Microsoft Windows SDK directory.

        Return
        ------
        str
            path
        """
    sdkdir = ''
    for ver in self.WindowsSdkVersion:
        loc = join(self.ri.windows_sdk, 'v%s' % ver)
        sdkdir = self.ri.lookup(loc, 'installationfolder')
        if sdkdir:
            break
    if not sdkdir or not isdir(sdkdir):
        path = join(self.ri.vc_for_python, '%0.1f' % self.vc_ver)
        install_base = self.ri.lookup(path, 'installdir')
        if install_base:
            sdkdir = join(install_base, 'WinSDK')
    if not sdkdir or not isdir(sdkdir):
        for ver in self.WindowsSdkVersion:
            intver = ver[:ver.rfind('.')]
            path = 'Microsoft SDKs\\Windows Kits\\%s' % intver
            d = join(self.ProgramFiles, path)
            if isdir(d):
                sdkdir = d
    if not sdkdir or not isdir(sdkdir):
        for ver in self.WindowsSdkVersion:
            path = 'Microsoft SDKs\\Windows\\v%s' % ver
            d = join(self.ProgramFiles, path)
            if isdir(d):
                sdkdir = d
    if not sdkdir:
        sdkdir = join(self.VCInstallDir, 'PlatformSDK')
    return sdkdir