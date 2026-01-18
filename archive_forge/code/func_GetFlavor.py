import errno
import filecmp
import os.path
import re
import tempfile
import sys
import subprocess
from collections.abc import MutableSet
def GetFlavor(params):
    """Returns |params.flavor| if it's set, the system's default flavor else."""
    flavors = {'cygwin': 'win', 'win32': 'win', 'darwin': 'mac'}
    if 'flavor' in params:
        return params['flavor']
    if sys.platform in flavors:
        return flavors[sys.platform]
    if sys.platform.startswith('sunos'):
        return 'solaris'
    if sys.platform.startswith(('dragonfly', 'freebsd')):
        return 'freebsd'
    if sys.platform.startswith('openbsd'):
        return 'openbsd'
    if sys.platform.startswith('netbsd'):
        return 'netbsd'
    if sys.platform.startswith('aix'):
        return 'aix'
    if sys.platform.startswith(('os390', 'zos')):
        return 'zos'
    return 'linux'