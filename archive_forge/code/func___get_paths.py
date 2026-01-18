from __future__ import annotations
import os
from ...git import (
from ...encoding import (
from ...util import (
from . import (
@staticmethod
def __get_paths(path: str) -> list[str]:
    """Return the list of available content paths under the given path."""
    git = Git(path)
    paths = git.get_file_names(['--cached', '--others', '--exclude-standard'])
    deleted_paths = git.get_file_names(['--deleted'])
    paths = sorted(set(paths) - set(deleted_paths))
    paths = [path + os.path.sep if os.path.isdir(to_bytes(path)) else path for path in paths]
    return paths