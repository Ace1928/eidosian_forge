import fnmatch
import io
import re
import tarfile
from functools import partial
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Union
from ray.data.block import BlockAccessor
from ray.data.datasource.file_based_datasource import FileBasedDatasource
from ray.util.annotations import PublicAPI
def _check_suffix(suffix: str, suffixes: Union[list, callable]):
    """Check whether a suffix is valid.

    Suffixes can be either None (=accept everything), a callable,
    or a list of patterns. If the pattern contains */? it is treated
    as a glob pattern, otherwise it is treated as a literal.

    Args:
        suffix: suffix to be checked
        suffixes: list of valid suffixes
    """
    if suffixes is None:
        return True
    if callable(suffixes):
        return suffixes(suffix)
    for pattern in suffixes:
        if '*' in pattern or '?' in pattern:
            if fnmatch.fnmatch('.' + suffix, pattern):
                return True
        elif suffix == pattern or '.' + suffix == pattern:
            return True
    return False