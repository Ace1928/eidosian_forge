import os
import subprocess
import winreg
from distutils.errors import DistutilsExecError, DistutilsPlatformError, \
from distutils.ccompiler import CCompiler, gen_lib_options
from distutils import log
from distutils.util import get_platform
from itertools import count
def _find_vc2015():
    try:
        key = winreg.OpenKeyEx(winreg.HKEY_LOCAL_MACHINE, 'Software\\Microsoft\\VisualStudio\\SxS\\VC7', access=winreg.KEY_READ | winreg.KEY_WOW64_32KEY)
    except OSError:
        log.debug('Visual C++ is not registered')
        return (None, None)
    best_version = 0
    best_dir = None
    with key:
        for i in count():
            try:
                v, vc_dir, vt = winreg.EnumValue(key, i)
            except OSError:
                break
            if v and vt == winreg.REG_SZ and os.path.isdir(vc_dir):
                try:
                    version = int(float(v))
                except (ValueError, TypeError):
                    continue
                if version >= 14 and version > best_version:
                    best_version, best_dir = (version, vc_dir)
    return (best_version, best_dir)