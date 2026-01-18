from __future__ import annotations
import gc
import logging
import os
import os.path as osp
from pathlib import Path
import re
import shlex
import warnings
import gitdb
from gitdb.db.loose import LooseObjectDB
from gitdb.exc import BadObject
from git.cmd import Git, handle_process_output
from git.compat import defenc, safe_decode
from git.config import GitConfigParser
from git.db import GitCmdObjectDB
from git.exc import (
from git.index import IndexFile
from git.objects import Submodule, RootModule, Commit
from git.refs import HEAD, Head, Reference, TagReference
from git.remote import Remote, add_progress, to_progress_instance
from git.util import (
from .fun import (
from git.types import (
from typing import (
from git.types import ConfigLevels_Tup, TypedDict
@staticmethod
def _to_full_tag_path(path: PathLike) -> str:
    path_str = str(path)
    if path_str.startswith(TagReference._common_path_default + '/'):
        return path_str
    if path_str.startswith(TagReference._common_default + '/'):
        return Reference._common_path_default + '/' + path_str
    else:
        return TagReference._common_path_default + '/' + path_str