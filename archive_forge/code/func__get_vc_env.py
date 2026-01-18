import os
import subprocess
import winreg
from distutils.errors import DistutilsExecError, DistutilsPlatformError, \
from distutils.ccompiler import CCompiler, gen_lib_options
from distutils import log
from distutils.util import get_platform
from itertools import count
def _get_vc_env(plat_spec):
    if os.getenv('DISTUTILS_USE_SDK'):
        return {key.lower(): value for key, value in os.environ.items()}
    vcvarsall, _ = _find_vcvarsall(plat_spec)
    if not vcvarsall:
        raise DistutilsPlatformError('Unable to find vcvarsall.bat')
    try:
        out = subprocess.check_output('cmd /u /c "{}" {} && set'.format(vcvarsall, plat_spec), stderr=subprocess.STDOUT).decode('utf-16le', errors='replace')
    except subprocess.CalledProcessError as exc:
        log.error(exc.output)
        raise DistutilsPlatformError('Error executing {}'.format(exc.cmd))
    env = {key.lower(): value for key, _, value in (line.partition('=') for line in out.splitlines()) if key and value}
    return env