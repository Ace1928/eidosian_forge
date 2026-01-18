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
def _config_parser(cls, repo: 'Repo', parent_commit: Union[Commit_ish, None], read_only: bool) -> SubmoduleConfigParser:
    """
        :return: Config Parser constrained to our submodule in read or write mode

        :raise IOError: If the .gitmodules file cannot be found, either locally or in
            the repository at the given parent commit. Otherwise the exception would be
            delayed until the first access of the config parser.
        """
    parent_matches_head = True
    if parent_commit is not None:
        try:
            parent_matches_head = repo.head.commit == parent_commit
        except ValueError:
            pass
    fp_module: Union[str, BytesIO]
    if not repo.bare and parent_matches_head and repo.working_tree_dir:
        fp_module = osp.join(repo.working_tree_dir, cls.k_modules_file)
    else:
        assert parent_commit is not None, 'need valid parent_commit in bare repositories'
        try:
            fp_module = cls._sio_modules(parent_commit)
        except KeyError as e:
            raise IOError('Could not find %s file in the tree of parent commit %s' % (cls.k_modules_file, parent_commit)) from e
    if not read_only and (repo.bare or not parent_matches_head):
        raise ValueError("Cannot write blobs of 'historical' submodule configurations")
    return SubmoduleConfigParser(fp_module, read_only=read_only)