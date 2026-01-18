import copy
import gyp.common
import os
import os.path
import re
import shlex
import subprocess
import sys
from gyp.common import GypError
def XcodeVersion():
    """Returns a tuple of version and build version of installed Xcode."""
    global XCODE_VERSION_CACHE
    if XCODE_VERSION_CACHE:
        return XCODE_VERSION_CACHE
    version = ''
    build = ''
    try:
        version_list = GetStdoutQuiet(['xcodebuild', '-version']).splitlines()
        if len(version_list) < 2:
            raise GypError('xcodebuild returned unexpected results')
        version = version_list[0].split()[-1]
        build = version_list[-1].split()[-1]
    except GypError:
        version = CLTVersion()
        if not version:
            raise GypError('No Xcode or CLT version detected!')
    version = version.split('.')[:3]
    version[0] = version[0].zfill(2)
    version = (''.join(version) + '00')[:4]
    XCODE_VERSION_CACHE = (version, build)
    return XCODE_VERSION_CACHE