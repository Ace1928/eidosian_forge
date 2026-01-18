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
def _find_git_files(dirname='', git_dir=None):
    """Behave like a file finder entrypoint plugin.

    We don't actually use the entrypoints system for this because it runs
    at absurd times. We only want to do this when we are building an sdist.
    """
    file_list = []
    if git_dir is None:
        git_dir = _run_git_functions()
    if git_dir:
        log.info('[pbr] In git context, generating filelist from git')
        file_list = _run_git_command(['ls-files', '-z'], git_dir)
        file_list = file_list.split(b'\x00'.decode('utf-8'))
    return [f for f in file_list if f]