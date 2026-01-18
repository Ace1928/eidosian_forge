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
def _iter_log_oneline(git_dir=None):
    """Iterate over --oneline log entries if possible.

    This parses the output into a structured form but does not apply
    presentation logic to the output - making it suitable for different
    uses.

    :return: An iterator of (hash, tags_set, 1st_line) tuples, or None if
        changelog generation is disabled / not available.
    """
    if git_dir is None:
        git_dir = _get_git_directory()
    if not git_dir:
        return []
    return _iter_log_inner(git_dir)