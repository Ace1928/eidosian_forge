import _imp
import _io
import sys
import _warnings
import marshal
@staticmethod
def _open_registry(key):
    try:
        return winreg.OpenKey(winreg.HKEY_CURRENT_USER, key)
    except OSError:
        return winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key)