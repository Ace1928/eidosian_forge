import os
import tempfile
from io import BytesIO
from itertools import chain
from ...objects import hex_to_sha
from ...repo import Repo, check_ref_format
from .utils import CompatTestCase, require_git_version, rmtree_ro, run_git_or_fail
def _parse_worktree_list(self, output):
    worktrees = []
    for line in BytesIO(output):
        fields = line.rstrip(b'\n').split()
        worktrees.append(tuple((f.decode() for f in fields)))
    return worktrees