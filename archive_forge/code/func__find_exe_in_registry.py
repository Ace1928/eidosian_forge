import os
import time
from platform import system
from subprocess import DEVNULL
from subprocess import STDOUT
from subprocess import Popen
from typing_extensions import deprecated
from selenium.common.exceptions import WebDriverException
from selenium.webdriver.common import utils
def _find_exe_in_registry(self):
    try:
        from _winreg import HKEY_CURRENT_USER
        from _winreg import HKEY_LOCAL_MACHINE
        from _winreg import OpenKey
        from _winreg import QueryValue
    except ImportError:
        from winreg import OpenKey, QueryValue, HKEY_LOCAL_MACHINE, HKEY_CURRENT_USER
    import shlex
    keys = ('SOFTWARE\\Classes\\FirefoxHTML\\shell\\open\\command', 'SOFTWARE\\Classes\\Applications\\firefox.exe\\shell\\open\\command')
    command = ''
    for path in keys:
        try:
            key = OpenKey(HKEY_LOCAL_MACHINE, path)
            command = QueryValue(key, '')
            break
        except OSError:
            try:
                key = OpenKey(HKEY_CURRENT_USER, path)
                command = QueryValue(key, '')
                break
            except OSError:
                pass
    else:
        return ''
    if not command:
        return ''
    return shlex.split(command)[0]