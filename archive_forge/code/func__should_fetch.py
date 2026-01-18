import logging
import os.path
import pathlib
import re
import urllib.parse
import urllib.request
from typing import List, Optional, Tuple
from pip._internal.exceptions import BadCommand, InstallationError
from pip._internal.utils.misc import HiddenText, display_path, hide_url
from pip._internal.utils.subprocess import make_command
from pip._internal.vcs.versioncontrol import (
@classmethod
def _should_fetch(cls, dest: str, rev: str) -> bool:
    """
        Return true if rev is a ref or is a commit that we don't have locally.

        Branches and tags are not considered in this method because they are
        assumed to be always available locally (which is a normal outcome of
        ``git clone`` and ``git fetch --tags``).
        """
    if rev.startswith('refs/'):
        return True
    if not looks_like_hash(rev):
        return False
    if cls.has_commit(dest, rev):
        return False
    return True