import abc
import configparser as cp
import fnmatch
from functools import wraps
import inspect
from io import BufferedReader, IOBase
import logging
import os
import os.path as osp
import re
import sys
from git.compat import defenc, force_text
from git.util import LockFile
from typing import (
from git.types import Lit_config_levels, ConfigLevels_Tup, PathLike, assert_never, _T
def _included_paths(self) -> List[Tuple[str, str]]:
    """List all paths that must be included to configuration.

        :return: The list of paths, where each path is a tuple of ``(option, value)``.
        """
    paths = []
    for section in self.sections():
        if section == 'include':
            paths += self.items(section)
        match = CONDITIONAL_INCLUDE_REGEXP.search(section)
        if match is None or self._repo is None:
            continue
        keyword = match.group(1)
        value = match.group(2).strip()
        if keyword in ['gitdir', 'gitdir/i']:
            value = osp.expanduser(value)
            if not any((value.startswith(s) for s in ['./', '/'])):
                value = '**/' + value
            if value.endswith('/'):
                value += '**'
            if keyword.endswith('/i'):
                value = re.sub('[a-zA-Z]', lambda m: '[{}{}]'.format(m.group().lower(), m.group().upper()), value)
            if self._repo.git_dir:
                if fnmatch.fnmatchcase(str(self._repo.git_dir), value):
                    paths += self.items(section)
        elif keyword == 'onbranch':
            try:
                branch_name = self._repo.active_branch.name
            except TypeError:
                continue
            if fnmatch.fnmatchcase(branch_name, value):
                paths += self.items(section)
    return paths