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
def _run_git_command(cmd, git_dir, **kwargs):
    if not isinstance(cmd, (list, tuple)):
        cmd = [cmd]
    return _run_shell_command(['git', '--git-dir=%s' % git_dir] + cmd, **kwargs)