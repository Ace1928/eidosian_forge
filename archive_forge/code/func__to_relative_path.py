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
@classmethod
def _to_relative_path(cls, parent_repo: 'Repo', path: PathLike) -> PathLike:
    """:return: a path guaranteed  to be relative to the given parent - repository
        :raise ValueError: if path is not contained in the parent repository's working tree"""
    path = to_native_path_linux(path)
    if path.endswith('/'):
        path = path[:-1]
    if osp.isabs(path) and parent_repo.working_tree_dir:
        working_tree_linux = to_native_path_linux(parent_repo.working_tree_dir)
        if not path.startswith(working_tree_linux):
            raise ValueError("Submodule checkout path '%s' needs to be within the parents repository at '%s'" % (working_tree_linux, path))
        path = path[len(working_tree_linux.rstrip('/')) + 1:]
        if not path:
            raise ValueError("Absolute submodule path '%s' didn't yield a valid relative path" % path)
    return path