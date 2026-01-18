import collections
import os
import re
import subprocess
import sys
from gyp.common import OrderedSet
import gyp.MSVSUtil
import gyp.MSVSVersion
def _FindDirectXInstallation():
    """Try to find an installation location for the DirectX SDK. Check for the
    standard environment variable, and if that doesn't exist, try to find
    via the registry. May return None if not found in either location."""
    if hasattr(_FindDirectXInstallation, 'dxsdk_dir'):
        return _FindDirectXInstallation.dxsdk_dir
    dxsdk_dir = os.environ.get('DXSDK_DIR')
    if not dxsdk_dir:
        cmd = ['reg.exe', 'query', 'HKLM\\Software\\Microsoft\\DirectX', '/s']
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout = p.communicate()[0].decode('utf-8')
        for line in stdout.splitlines():
            if 'InstallPath' in line:
                dxsdk_dir = line.split('    ')[3] + '\\'
    _FindDirectXInstallation.dxsdk_dir = dxsdk_dir
    return dxsdk_dir