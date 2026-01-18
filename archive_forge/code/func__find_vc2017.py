import os
import subprocess
import winreg
from distutils.errors import DistutilsExecError, DistutilsPlatformError, \
from distutils.ccompiler import CCompiler, gen_lib_options
from distutils import log
from distutils.util import get_platform
from itertools import count
def _find_vc2017():
    """Returns "15, path" based on the result of invoking vswhere.exe
    If no install is found, returns "None, None"

    The version is returned to avoid unnecessarily changing the function
    result. It may be ignored when the path is not None.

    If vswhere.exe is not available, by definition, VS 2017 is not
    installed.
    """
    root = os.environ.get('ProgramFiles(x86)') or os.environ.get('ProgramFiles')
    if not root:
        return (None, None)
    try:
        path = subprocess.check_output([os.path.join(root, 'Microsoft Visual Studio', 'Installer', 'vswhere.exe'), '-latest', '-prerelease', '-requires', 'Microsoft.VisualStudio.Component.VC.Tools.x86.x64', '-property', 'installationPath', '-products', '*'], encoding='mbcs', errors='strict').strip()
    except (subprocess.CalledProcessError, OSError, UnicodeDecodeError):
        return (None, None)
    path = os.path.join(path, 'VC', 'Auxiliary', 'Build')
    if os.path.isdir(path):
        return (15, path)
    return (None, None)