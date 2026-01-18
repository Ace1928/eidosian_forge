from gitdb.exc import (
from git.compat import safe_decode
from git.util import remove_password_if_present
from typing import List, Sequence, Tuple, Union, TYPE_CHECKING
from git.types import PathLike
class CheckoutError(GitError):
    """Thrown if a file could not be checked out from the index as it contained
    changes.

    The :attr:`failed_files` attribute contains a list of relative paths that failed to
    be checked out as they contained changes that did not exist in the index.

    The :attr:`failed_reasons` attribute contains a string informing about the actual
    cause of the issue.

    The :attr:`valid_files` attribute contains a list of relative paths to files that
    were checked out successfully and hence match the version stored in the index.
    """

    def __init__(self, message: str, failed_files: Sequence[PathLike], valid_files: Sequence[PathLike], failed_reasons: List[str]) -> None:
        Exception.__init__(self, message)
        self.failed_files = failed_files
        self.failed_reasons = failed_reasons
        self.valid_files = valid_files

    def __str__(self) -> str:
        return Exception.__str__(self) + ':%s' % self.failed_files