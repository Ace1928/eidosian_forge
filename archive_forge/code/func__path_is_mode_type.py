import _imp
import _io
import sys
import _warnings
import marshal
def _path_is_mode_type(path, mode):
    """Test whether the path is the specified mode type."""
    try:
        stat_info = _path_stat(path)
    except OSError:
        return False
    return stat_info.st_mode & 61440 == mode