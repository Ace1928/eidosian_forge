import errno
import os
import re
import subprocess
import sys
import glob
def _RegistryGetValue(key, value):
    """Use _winreg or reg.exe to obtain the value of a registry key.

  Using _winreg is preferable because it solves an issue on some corporate
  environments where access to reg.exe is locked down. However, we still need
  to fallback to reg.exe for the case where the _winreg module is not available
  (for example in cygwin python).

  Args:
    key: The registry key.
    value: The particular registry value to read.
  Return:
    contents of the registry key's value, or None on failure.
  """
    try:
        return _RegistryGetValueUsingWinReg(key, value)
    except ImportError:
        pass
    text = _RegistryQuery(key, value)
    if not text:
        return None
    match = re.search('REG_\\w+\\s+([^\\r]+)\\r\\n', text)
    if not match:
        return None
    return match.group(1)