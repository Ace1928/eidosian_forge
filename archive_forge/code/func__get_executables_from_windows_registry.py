import os
import sys
import hashlib
import filecmp
from collections import namedtuple
from shutil import which
from jedi.cache import memoize_method, time_cache
from jedi.inference.compiled.subprocess import CompiledSubprocess, \
import parso
def _get_executables_from_windows_registry(version):
    import winreg
    sub_keys = ['SOFTWARE\\Python\\PythonCore\\{version}\\InstallPath', 'SOFTWARE\\Wow6432Node\\Python\\PythonCore\\{version}\\InstallPath', 'SOFTWARE\\Python\\PythonCore\\{version}-32\\InstallPath', 'SOFTWARE\\Wow6432Node\\Python\\PythonCore\\{version}-32\\InstallPath']
    for root_key in [winreg.HKEY_CURRENT_USER, winreg.HKEY_LOCAL_MACHINE]:
        for sub_key in sub_keys:
            sub_key = sub_key.format(version=version)
            try:
                with winreg.OpenKey(root_key, sub_key) as key:
                    prefix = winreg.QueryValueEx(key, '')[0]
                    exe = os.path.join(prefix, 'python.exe')
                    if os.path.isfile(exe):
                        yield exe
            except WindowsError:
                pass