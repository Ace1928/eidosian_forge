from gitdb.exc import (
from git.compat import safe_decode
from git.util import remove_password_if_present
from typing import List, Sequence, Tuple, Union, TYPE_CHECKING
from git.types import PathLike
class HookExecutionError(CommandError):
    """Thrown if a hook exits with a non-zero exit code.

    This provides access to the exit code and the string returned via standard output.
    """

    def __init__(self, command: Union[List[str], Tuple[str, ...], str], status: Union[str, int, None, Exception], stderr: Union[bytes, str, None]=None, stdout: Union[bytes, str, None]=None) -> None:
        super().__init__(command, status, stderr, stdout)
        self._msg = "Hook('%s') failed%s"