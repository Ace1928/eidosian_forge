from __future__ import unicode_literals
import distutils.errors
from distutils import log
import errno
import io
import os
import re
import subprocess
import time
import pkg_resources
from pbr import options
from pbr import version
def _get_git_directory():
    try:
        return _run_shell_command(['git', 'rev-parse', '--git-dir'])
    except OSError as e:
        if e.errno == errno.ENOENT:
            return ''
        raise