import fnmatch
import io
import re
import tarfile
from functools import partial
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Union
from ray.data.block import BlockAccessor
from ray.data.datasource.file_based_datasource import FileBasedDatasource
from ray.util.annotations import PublicAPI
def _base_plus_ext(path: str):
    """Split off all file extensions.

    Returns base, allext.

    Args:
        path: path with extensions

    Returns:
        str: path with all extensions removed
    """
    match = re.match('^((?:.*/|)[^.]+)[.]([^/]*)$', path)
    if not match:
        return (None, None)
    return (match.group(1), match.group(2))