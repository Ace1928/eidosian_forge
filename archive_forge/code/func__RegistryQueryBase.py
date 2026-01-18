import errno
import os
import re
import subprocess
import sys
import glob
def _RegistryQueryBase(sysdir, key, value):
    """Use reg.exe to read a particular key.

  While ideally we might use the win32 module, we would like gyp to be
  python neutral, so for instance cygwin python lacks this module.

  Arguments:
    sysdir: The system subdirectory to attempt to launch reg.exe from.
    key: The registry key to read from.
    value: The particular value to read.
  Return:
    stdout from reg.exe, or None for failure.
  """
    if sys.platform not in ('win32', 'cygwin'):
        return None
    cmd = [os.path.join(os.environ.get('WINDIR', ''), sysdir, 'reg.exe'), 'query', key]
    if value:
        cmd.extend(['/v', value])
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    text = p.communicate()[0].decode('utf-8')
    if p.returncode:
        return None
    return text