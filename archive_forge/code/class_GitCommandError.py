from gitdb.exc import (
from git.compat import safe_decode
from git.util import remove_password_if_present
from typing import List, Sequence, Tuple, Union, TYPE_CHECKING
from git.types import PathLike
class GitCommandError(CommandError):
    """Thrown if execution of the git command fails with non-zero status code."""

    def __init__(self, command: Union[List[str], Tuple[str, ...], str], status: Union[str, int, None, Exception]=None, stderr: Union[bytes, str, None]=None, stdout: Union[bytes, str, None]=None) -> None:
        super().__init__(command, status, stderr, stdout)