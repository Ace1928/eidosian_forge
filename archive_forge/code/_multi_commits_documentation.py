import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Iterable, List, Optional, Set, Tuple, Union
from ._commit_api import CommitOperationAdd, CommitOperationDelete
from .community import DiscussionWithDetails
from .utils import experimental
from .utils._cache_manager import _format_size
from .utils.insecure_hashlib import sha256
Format a step for PR description.

        Formatting can be changed in the future as long as it is single line, starts with `- [ ]`/`- [x]` and contains
        `self.id`. Must be able to match `STEP_ID_REGEX`.
        