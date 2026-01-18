from __future__ import annotations
import os
import subprocess
import json
import pathlib
import shutil
import tempfile
import locale
from .. import mlog
from .core import MesonException
from .universal import is_windows, windows_detect_native_arch
def _setup_vsenv(force: bool) -> bool:
    if not is_windows():
        return False
    if os.environ.get('OSTYPE') == 'cygwin':
        return False
    if 'MESON_FORCE_VSENV_FOR_UNITTEST' not in os.environ:
        if 'VSINSTALLDIR' in os.environ:
            return False
        if shutil.which('cl.exe'):
            return False
    if not force:
        if shutil.which('cc'):
            return False
        if shutil.which('gcc'):
            return False
        if shutil.which('clang'):
            return False
        if shutil.which('clang-cl'):
            return False
    root = os.environ.get('ProgramFiles(x86)') or os.environ.get('ProgramFiles')
    bat_locator_bin = pathlib.Path(root, 'Microsoft Visual Studio/Installer/vswhere.exe')
    if not bat_locator_bin.exists():
        raise MesonException(f'Could not find {bat_locator_bin}')
    bat_json = subprocess.check_output([str(bat_locator_bin), '-latest', '-prerelease', '-requiresAny', '-requires', 'Microsoft.VisualStudio.Component.VC.Tools.x86.x64', '-requires', 'Microsoft.VisualStudio.Workload.WDExpress', '-products', '*', '-utf8', '-format', 'json'])
    bat_info = json.loads(bat_json)
    if not bat_info:
        raise MesonException('Could not parse vswhere.exe output')
    bat_root = pathlib.Path(bat_info[0]['installationPath'])
    if windows_detect_native_arch() == 'arm64':
        bat_path = bat_root / 'VC/Auxiliary/Build/vcvarsarm64.bat'
        if not bat_path.exists():
            bat_path = bat_root / 'VC/Auxiliary/Build/vcvarsx86_arm64.bat'
    else:
        bat_path = bat_root / 'VC/Auxiliary/Build/vcvars64.bat'
        if not bat_path.exists():
            bat_path = bat_root / 'VC/Auxiliary/Build/vcvarsx86_amd64.bat'
    if not bat_path.exists():
        raise MesonException(f'Could not find {bat_path}')
    mlog.log('Activating VS', bat_info[0]['catalog']['productDisplayVersion'])
    bat_separator = '---SPLIT---'
    bat_contents = bat_template.format(bat_path, bat_separator)
    bat_file = tempfile.NamedTemporaryFile('w', suffix='.bat', encoding='utf-8', delete=False)
    bat_file.write(bat_contents)
    bat_file.flush()
    bat_file.close()
    bat_output = subprocess.check_output(bat_file.name, universal_newlines=True, encoding=locale.getpreferredencoding(False))
    os.unlink(bat_file.name)
    bat_lines = bat_output.split('\n')
    bat_separator_seen = False
    for bat_line in bat_lines:
        if bat_line == bat_separator:
            bat_separator_seen = True
            continue
        if not bat_separator_seen:
            continue
        if not bat_line:
            continue
        try:
            k, v = bat_line.split('=', 1)
        except ValueError:
            pass
        else:
            os.environ[k] = v
    return True