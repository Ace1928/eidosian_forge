import gyp.common
import json
import os
import posixpath
def _ToLocalPath(toplevel_dir, path):
    """Converts |path| to a path relative to |toplevel_dir|."""
    if path == toplevel_dir:
        return ''
    if path.startswith(toplevel_dir + '/'):
        return path[len(toplevel_dir) + len('/'):]
    return path