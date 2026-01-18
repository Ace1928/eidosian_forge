import functools
import os
import sys
import sysconfig
from importlib.util import cache_from_source
from typing import Any, Callable, Dict, Generator, Iterable, List, Optional, Set, Tuple
from pip._internal.exceptions import UninstallationError
from pip._internal.locations import get_bin_prefix, get_bin_user
from pip._internal.metadata import BaseDistribution
from pip._internal.utils.compat import WINDOWS
from pip._internal.utils.egg_link import egg_link_path_from_location
from pip._internal.utils.logging import getLogger, indent_log
from pip._internal.utils.misc import ask, normalize_path, renames, rmtree
from pip._internal.utils.temp_dir import AdjacentTempDirectory, TempDirectory
from pip._internal.utils.virtualenv import running_under_virtualenv
def _get_file_stash(self, path: str) -> str:
    """Stashes a file.

        If no root has been provided, one will be created for the directory
        in the user's temp directory."""
    path = os.path.normcase(path)
    head, old_head = (os.path.dirname(path), None)
    save_dir = None
    while head != old_head:
        try:
            save_dir = self._save_dirs[head]
            break
        except KeyError:
            pass
        head, old_head = (os.path.dirname(head), head)
    else:
        head = os.path.dirname(path)
        save_dir = TempDirectory(kind='uninstall')
        self._save_dirs[head] = save_dir
    relpath = os.path.relpath(path, head)
    if relpath and relpath != os.path.curdir:
        return os.path.join(save_dir.path, relpath)
    return save_dir.path