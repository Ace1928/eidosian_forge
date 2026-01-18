import os
import warnings
from contextlib import suppress
from typing import Any, Dict, Optional, Set
from .errors import PackedRefsException, RefFormatError
from .file import GitFile, ensure_dir_exists
from .objects import ZERO_SHA, ObjectID, Tag, git_line, valid_hexsha
from .pack import ObjectContainer
def _check_refname(self, name):
    """Ensure a refname is valid and lives in refs or is HEAD.

        HEAD is not a valid refname according to git-check-ref-format, but this
        class needs to be able to touch HEAD. Also, check_ref_format expects
        refnames without the leading 'refs/', but this class requires that
        so it cannot touch anything outside the refs dir (or HEAD).

        Args:
          name: The name of the reference.

        Raises:
          KeyError: if a refname is not HEAD or is otherwise not valid.
        """
    if name in (HEADREF, b'refs/stash'):
        return
    if not name.startswith(b'refs/') or not check_ref_format(name[5:]):
        raise RefFormatError(name)