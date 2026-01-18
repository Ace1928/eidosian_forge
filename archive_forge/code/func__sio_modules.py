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
def _sio_modules(cls, parent_commit: Commit_ish) -> BytesIO:
    """:return: Configuration file as BytesIO - we only access it through the respective blob's data"""
    sio = BytesIO(parent_commit.tree[cls.k_modules_file].data_stream.read())
    sio.name = cls.k_modules_file
    return sio