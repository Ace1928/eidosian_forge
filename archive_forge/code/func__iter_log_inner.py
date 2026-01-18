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
def _iter_log_inner(git_dir):
    """Iterate over --oneline log entries.

    This parses the output intro a structured form but does not apply
    presentation logic to the output - making it suitable for different
    uses.

    .. caution:: this function risk to return a tag that doesn't exist really
                 inside the git objects list due to replacement made
                 to tag name to also list pre-release suffix.
                 Compliant with the SemVer specification (e.g 1.2.3-rc1)

    :return: An iterator of (hash, tags_set, 1st_line) tuples.
    """
    log.info('[pbr] Generating ChangeLog')
    log_cmd = ['log', '--decorate=full', '--format=%h%x00%s%x00%d']
    changelog = _run_git_command(log_cmd, git_dir)
    for line in changelog.split('\n'):
        line_parts = line.split('\x00')
        if len(line_parts) != 3:
            continue
        sha, msg, refname = line_parts
        tags = set()
        if 'refs/tags/' in refname:
            refname = refname.strip()[1:-1]
            for tag_string in refname.split('refs/tags/')[1:]:
                candidate = tag_string.split(', ')[0].replace('-', '.')
                if _is_valid_version(candidate):
                    tags.add(candidate)
        yield (sha, tags, msg)