import _imp
import _io
import sys
import _warnings
import marshal
def _path_isabs(path):
    """Replacement for os.path.isabs."""
    return path.startswith(path_separators)