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
def _config_reader(self, config_level: Optional[Lit_config_levels]=None, git_dir: Optional[PathLike]=None) -> GitConfigParser:
    if config_level is None:
        files = [self._get_config_path(cast(Lit_config_levels, f), git_dir) for f in self.config_level if cast(Lit_config_levels, f)]
    else:
        files = [self._get_config_path(config_level, git_dir)]
    return GitConfigParser(files, read_only=True, repo=self)