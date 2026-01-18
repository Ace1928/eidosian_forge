import collections
import os
import re
import sys
import functools
import itertools
def _syscmd_ver(system='', release='', version='', supported_platforms=('win32', 'win16', 'dos')):
    """ Tries to figure out the OS version used and returns
        a tuple (system, release, version).

        It uses the "ver" shell command for this which is known
        to exists on Windows, DOS. XXX Others too ?

        In case this fails, the given parameters are used as
        defaults.

    """
    if sys.platform not in supported_platforms:
        return (system, release, version)
    import subprocess
    for cmd in ('ver', 'command /c ver', 'cmd /c ver'):
        try:
            info = subprocess.check_output(cmd, stdin=subprocess.DEVNULL, stderr=subprocess.DEVNULL, text=True, encoding='locale', shell=True)
        except (OSError, subprocess.CalledProcessError) as why:
            continue
        else:
            break
    else:
        return (system, release, version)
    info = info.strip()
    m = _ver_output.match(info)
    if m is not None:
        system, release, version = m.groups()
        if release[-1] == '.':
            release = release[:-1]
        if version[-1] == '.':
            version = version[:-1]
        version = _norm_version(version)
    return (system, release, version)