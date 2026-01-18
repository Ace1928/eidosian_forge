import gyp.common
import json
import os
import posixpath
def _ToGypPath(path):
    """Converts a path to the format used by gyp."""
    if os.sep == '\\' and os.altsep == '/':
        return path.replace('\\', '/')
    return path