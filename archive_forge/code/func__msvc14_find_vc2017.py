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
def _msvc14_find_vc2017():
    """Python 3.8 "distutils/_msvccompiler.py" backport

    Returns "15, path" based on the result of invoking vswhere.exe
    If no install is found, returns "None, None"

    The version is returned to avoid unnecessarily changing the function
    result. It may be ignored when the path is not None.

    If vswhere.exe is not available, by definition, VS 2017 is not
    installed.
    """
    root = environ.get('ProgramFiles(x86)') or environ.get('ProgramFiles')
    if not root:
        return (None, None)
    suitable_components = ('Microsoft.VisualStudio.Component.VC.Tools.x86.x64', 'Microsoft.VisualStudio.Workload.WDExpress')
    for component in suitable_components:
        with contextlib.suppress(CalledProcessError, OSError, UnicodeDecodeError):
            path = subprocess.check_output([join(root, 'Microsoft Visual Studio', 'Installer', 'vswhere.exe'), '-latest', '-prerelease', '-requires', component, '-property', 'installationPath', '-products', '*']).decode(encoding='mbcs', errors='strict').strip()
            path = join(path, 'VC', 'Auxiliary', 'Build')
            if isdir(path):
                return (15, path)
    return (None, None)