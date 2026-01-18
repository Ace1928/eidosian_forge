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
def _get_raw_tag_info(git_dir):
    describe = _run_git_command(['describe', '--always'], git_dir)
    if '-' in describe:
        return describe.rsplit('-', 2)[-2]
    if '.' in describe:
        return 0
    return None