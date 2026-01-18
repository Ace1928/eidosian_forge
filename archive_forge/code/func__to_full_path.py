from stat import S_ISDIR
from git.compat import safe_decode, defenc
from typing import (
def _to_full_path(item: EntryTupOrNone, path_prefix: str) -> EntryTupOrNone:
    """Rebuild entry with given path prefix."""
    if not item:
        return item
    return (item[0], item[1], path_prefix + item[2])