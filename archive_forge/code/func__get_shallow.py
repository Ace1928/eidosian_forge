import errno
import os
import shutil
import socket
import tempfile
from ...objects import hex_to_sha
from ...protocol import CAPABILITY_SIDE_BAND_64K
from ...repo import Repo
from ...server import ReceivePackHandler
from ..utils import tear_down_repo
from .utils import require_git_version, run_git_or_fail
def _get_shallow(repo):
    shallow_file = repo.get_named_file('shallow')
    if not shallow_file:
        return []
    shallows = []
    with shallow_file:
        for line in shallow_file:
            sha = line.strip()
            if not sha:
                continue
            hex_to_sha(sha)
            shallows.append(sha)
    return shallows