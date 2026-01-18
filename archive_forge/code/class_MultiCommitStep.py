import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Iterable, List, Optional, Set, Tuple, Union
from ._commit_api import CommitOperationAdd, CommitOperationDelete
from .community import DiscussionWithDetails
from .utils import experimental
from .utils._cache_manager import _format_size
from .utils.insecure_hashlib import sha256
@dataclass
class MultiCommitStep:
    """Dataclass containing a list of CommitOperation to commit at once.

    A [`MultiCommitStep`] is one atomic part of a [`MultiCommitStrategy`]. Each step is identified by its own
    deterministic ID based on the list of commit operations (hexadecimal sha256). ID is persistent between re-runs if
    the list of commits is kept the same.
    """
    operations: List[Union[CommitOperationAdd, CommitOperationDelete]]
    id: str = field(init=False)
    completed: bool = False

    def __post_init__(self) -> None:
        if len(self.operations) == 0:
            raise ValueError('A MultiCommitStep must have at least 1 commit operation, got 0.')
        sha = sha256()
        for op in self.operations:
            if isinstance(op, CommitOperationAdd):
                sha.update(b'ADD')
                sha.update(op.path_in_repo.encode())
                sha.update(op.upload_info.sha256)
            elif isinstance(op, CommitOperationDelete):
                sha.update(b'DELETE')
                sha.update(op.path_in_repo.encode())
                sha.update(str(op.is_folder).encode())
            else:
                NotImplementedError()
        self.id = sha.hexdigest()

    def __str__(self) -> str:
        """Format a step for PR description.

        Formatting can be changed in the future as long as it is single line, starts with `- [ ]`/`- [x]` and contains
        `self.id`. Must be able to match `STEP_ID_REGEX`.
        """
        additions = [op for op in self.operations if isinstance(op, CommitOperationAdd)]
        file_deletions = [op for op in self.operations if isinstance(op, CommitOperationDelete) and (not op.is_folder)]
        folder_deletions = [op for op in self.operations if isinstance(op, CommitOperationDelete) and op.is_folder]
        if len(additions) > 0:
            return f'- [{('x' if self.completed else ' ')}] Upload {len(additions)} file(s) totalling {_format_size(sum((add.upload_info.size for add in additions)))} ({self.id})'
        else:
            return f'- [{('x' if self.completed else ' ')}] Delete {len(file_deletions)} file(s) and {len(folder_deletions)} folder(s) ({self.id})'