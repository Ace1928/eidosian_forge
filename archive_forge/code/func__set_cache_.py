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
def _set_cache_(self, attr: str) -> None:
    if attr in ('path', '_url', '_branch_path'):
        reader: SectionConstraint = self.config_reader()
        try:
            self.path = reader.get('path')
        except cp.NoSectionError as e:
            if self.repo.working_tree_dir is not None:
                raise ValueError("This submodule instance does not exist anymore in '%s' file" % osp.join(self.repo.working_tree_dir, '.gitmodules')) from e
        self._url = reader.get('url')
        self._branch_path = reader.get_value(self.k_head_option, git.Head.to_full_path(self.k_head_default))
    elif attr == '_name':
        raise AttributeError('Cannot retrieve the name of a submodule if it was not set initially')
    else:
        super()._set_cache_(attr)