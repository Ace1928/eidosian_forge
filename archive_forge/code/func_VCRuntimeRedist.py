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
def VCRuntimeRedist(self):
    """
        Microsoft Visual C++ runtime redistributable dll.

        Return
        ------
        str
            path
        """
    vcruntime = 'vcruntime%d0.dll' % self.vc_ver
    arch_subdir = self.pi.target_dir(x64=True).strip('\\')
    prefixes = []
    tools_path = self.si.VCInstallDir
    redist_path = dirname(tools_path.replace('\\Tools', '\\Redist'))
    if isdir(redist_path):
        redist_path = join(redist_path, listdir(redist_path)[-1])
        prefixes += [redist_path, join(redist_path, 'onecore')]
    prefixes += [join(tools_path, 'redist')]
    crt_dirs = ('Microsoft.VC%d.CRT' % (self.vc_ver * 10), 'Microsoft.VC%d.CRT' % (int(self.vs_ver) * 10))
    for prefix, crt_dir in itertools.product(prefixes, crt_dirs):
        path = join(prefix, arch_subdir, crt_dir, vcruntime)
        if isfile(path):
            return path