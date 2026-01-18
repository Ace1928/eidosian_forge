import os
import stat
import sys
import time
import warnings
from io import BytesIO
from typing import (
from .errors import (
from .file import GitFile
from .hooks import (
from .line_ending import BlobNormalizer, TreeBlobNormalizer
from .object_store import (
from .objects import (
from .pack import generate_unpacked_objects
from .refs import (
@classmethod
def _init_new_working_directory(cls, path, main_repo, identifier=None, mkdir=False):
    """Create a new working directory linked to a repository.

        Args:
          path: Path in which to create the working tree.
          main_repo: Main repository to reference
          identifier: Worktree identifier
          mkdir: Whether to create the directory
        Returns: `Repo` instance
        """
    if mkdir:
        os.mkdir(path)
    if identifier is None:
        identifier = os.path.basename(path)
    main_worktreesdir = os.path.join(main_repo.controldir(), WORKTREES)
    worktree_controldir = os.path.join(main_worktreesdir, identifier)
    gitdirfile = os.path.join(path, CONTROLDIR)
    with open(gitdirfile, 'wb') as f:
        f.write(b'gitdir: ' + os.fsencode(worktree_controldir) + b'\n')
    try:
        os.mkdir(main_worktreesdir)
    except FileExistsError:
        pass
    try:
        os.mkdir(worktree_controldir)
    except FileExistsError:
        pass
    with open(os.path.join(worktree_controldir, GITDIR), 'wb') as f:
        f.write(os.fsencode(gitdirfile) + b'\n')
    with open(os.path.join(worktree_controldir, COMMONDIR), 'wb') as f:
        f.write(b'../..\n')
    with open(os.path.join(worktree_controldir, 'HEAD'), 'wb') as f:
        f.write(main_repo.head() + b'\n')
    r = cls(path)
    r.reset_index()
    return r