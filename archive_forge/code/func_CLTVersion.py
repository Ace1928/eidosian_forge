import copy
import gyp.common
import os
import os.path
import re
import shlex
import subprocess
import sys
from gyp.common import GypError
def CLTVersion():
    """Returns the version of command-line tools from pkgutil."""
    STANDALONE_PKG_ID = 'com.apple.pkg.DeveloperToolsCLILeo'
    FROM_XCODE_PKG_ID = 'com.apple.pkg.DeveloperToolsCLI'
    MAVERICKS_PKG_ID = 'com.apple.pkg.CLTools_Executables'
    regex = re.compile('version: (?P<version>.+)')
    for key in [MAVERICKS_PKG_ID, STANDALONE_PKG_ID, FROM_XCODE_PKG_ID]:
        try:
            output = GetStdout(['/usr/sbin/pkgutil', '--pkg-info', key])
            return re.search(regex, output).groupdict()['version']
        except GypError:
            continue
    regex = re.compile('Command Line Tools for Xcode\\s+(?P<version>\\S+)')
    try:
        output = GetStdout(['/usr/sbin/softwareupdate', '--history'])
        return re.search(regex, output).groupdict()['version']
    except GypError:
        return None