import _imp
import _io
import sys
import _warnings
import marshal
def _path_isfile(path):
    """Replacement for os.path.isfile."""
    return _path_is_mode_type(path, 32768)