import os
import platform
import pathlib
def _data_root_Windows():
    release, version, csd, ptype = platform.win32_ver()
    root = _settings_root_XP() if release == 'XP' else _settings_root_Vista()
    return pathlib.Path(root, 'Python Keyring')