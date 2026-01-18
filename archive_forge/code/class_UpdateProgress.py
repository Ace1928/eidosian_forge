import gc
from io import BytesIO
import logging
import os
import os.path as osp
import stat
import uuid
import git
from git.cmd import Git
from git.compat import defenc
from git.config import GitConfigParser, SectionConstraint, cp
from git.exc import (
from git.objects.base import IndexObject, Object
from git.objects.util import TraversableIterableObj
from git.util import (
from .util import (
from typing import Callable, Dict, Mapping, Sequence, TYPE_CHECKING, cast
from typing import Any, Iterator, Union
from git.types import Commit_ish, Literal, PathLike, TBD
class UpdateProgress(RemoteProgress):
    """Class providing detailed progress information to the caller who should
    derive from it and implement the ``update(...)`` message."""
    CLONE, FETCH, UPDWKTREE = [1 << x for x in range(RemoteProgress._num_op_codes, RemoteProgress._num_op_codes + 3)]
    _num_op_codes: int = RemoteProgress._num_op_codes + 3
    __slots__ = ()