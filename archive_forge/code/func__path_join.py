import _imp
import _io
import sys
import _warnings
import marshal
def _path_join(*path_parts):
    """Replacement for os.path.join()."""
    return path_sep.join([part.rstrip(path_separators) for part in path_parts if part])