from .base import Submodule, UpdateProgress
from .util import find_first_remote_branch
from git.exc import InvalidGitRepositoryError
import git
import logging
from typing import TYPE_CHECKING, Union
from git.types import Commit_ish
class RootUpdateProgress(UpdateProgress):
    """Utility class which adds more opcodes to the UpdateProgress."""
    REMOVE, PATHCHANGE, BRANCHCHANGE, URLCHANGE = [1 << x for x in range(UpdateProgress._num_op_codes, UpdateProgress._num_op_codes + 4)]
    _num_op_codes = UpdateProgress._num_op_codes + 4
    __slots__ = ()