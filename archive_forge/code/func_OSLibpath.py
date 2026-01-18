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
def OSLibpath(self):
    """
        Microsoft Windows SDK Libraries Paths.

        Return
        ------
        list of str
            paths
        """
    ref = join(self.si.WindowsSdkDir, 'References')
    libpath = []
    if self.vs_ver <= 9.0:
        libpath += self.OSLibraries
    if self.vs_ver >= 11.0:
        libpath += [join(ref, 'CommonConfiguration\\Neutral')]
    if self.vs_ver >= 14.0:
        libpath += [ref, join(self.si.WindowsSdkDir, 'UnionMetadata'), join(ref, 'Windows.Foundation.UniversalApiContract', '1.0.0.0'), join(ref, 'Windows.Foundation.FoundationContract', '1.0.0.0'), join(ref, 'Windows.Networking.Connectivity.WwanContract', '1.0.0.0'), join(self.si.WindowsSdkDir, 'ExtensionSDKs', 'Microsoft.VCLibs', '%0.1f' % self.vs_ver, 'References', 'CommonConfiguration', 'neutral')]
    return libpath