from abc import abstractmethod
import contextlib
from functools import wraps
import getpass
import logging
import os
import os.path as osp
import pathlib
import platform
import re
import shutil
import stat
import subprocess
import sys
import time
from urllib.parse import urlsplit, urlunsplit
import warnings
from typing import (
from .types import (
from gitdb.util import (  # noqa: F401  # @IgnorePep8
@classmethod
def _list_from_string(cls, repo: 'Repo', text: str) -> 'Stats':
    """Create a Stat object from output retrieved by git-diff.

        :return: git.Stat
        """
    hsh: HSH_TD = {'total': {'insertions': 0, 'deletions': 0, 'lines': 0, 'files': 0}, 'files': {}}
    for line in text.splitlines():
        raw_insertions, raw_deletions, filename = line.split('\t')
        insertions = raw_insertions != '-' and int(raw_insertions) or 0
        deletions = raw_deletions != '-' and int(raw_deletions) or 0
        hsh['total']['insertions'] += insertions
        hsh['total']['deletions'] += deletions
        hsh['total']['lines'] += insertions + deletions
        hsh['total']['files'] += 1
        files_dict: Files_TD = {'insertions': insertions, 'deletions': deletions, 'lines': insertions + deletions}
        hsh['files'][filename.strip()] = files_dict
    return Stats(hsh['total'], hsh['files'])