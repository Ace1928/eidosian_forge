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
def _msvc14_get_vc_env(plat_spec):
    """Python 3.8 "distutils/_msvccompiler.py" backport"""
    if 'DISTUTILS_USE_SDK' in environ:
        return {key.lower(): value for key, value in environ.items()}
    vcvarsall, vcruntime = _msvc14_find_vcvarsall(plat_spec)
    if not vcvarsall:
        raise distutils.errors.DistutilsPlatformError('Unable to find vcvarsall.bat')
    try:
        out = subprocess.check_output('cmd /u /c "{}" {} && set'.format(vcvarsall, plat_spec), stderr=subprocess.STDOUT).decode('utf-16le', errors='replace')
    except subprocess.CalledProcessError as exc:
        raise distutils.errors.DistutilsPlatformError('Error executing {}'.format(exc.cmd)) from exc
    env = {key.lower(): value for key, _, value in (line.partition('=') for line in out.splitlines()) if key and value}
    if vcruntime:
        env['py_vcruntime_redist'] = vcruntime
    return env