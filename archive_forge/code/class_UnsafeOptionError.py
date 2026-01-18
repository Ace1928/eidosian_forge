from gitdb.exc import (
from git.compat import safe_decode
from git.util import remove_password_if_present
from typing import List, Sequence, Tuple, Union, TYPE_CHECKING
from git.types import PathLike
class UnsafeOptionError(GitError):
    """Thrown if unsafe options are passed without being explicitly allowed."""