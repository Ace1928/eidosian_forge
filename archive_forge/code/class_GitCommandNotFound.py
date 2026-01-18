from gitdb.exc import (
from git.compat import safe_decode
from git.util import remove_password_if_present
from typing import List, Sequence, Tuple, Union, TYPE_CHECKING
from git.types import PathLike
class GitCommandNotFound(CommandError):
    """Thrown if we cannot find the `git` executable in the PATH or at the path given by
    the GIT_PYTHON_GIT_EXECUTABLE environment variable."""

    def __init__(self, command: Union[List[str], Tuple[str], str], cause: Union[str, Exception]) -> None:
        super().__init__(command, cause)
        self._msg = "Cmd('%s') not found%s"