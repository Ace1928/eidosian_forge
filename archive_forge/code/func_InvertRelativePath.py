import errno
import filecmp
import os.path
import re
import tempfile
import sys
import subprocess
from collections.abc import MutableSet
@memoize
def InvertRelativePath(path, toplevel_dir=None):
    """Given a path like foo/bar that is relative to toplevel_dir, return
  the inverse relative path back to the toplevel_dir.

  E.g. os.path.normpath(os.path.join(path, InvertRelativePath(path)))
  should always produce the empty string, unless the path contains symlinks.
  """
    if not path:
        return path
    toplevel_dir = '.' if toplevel_dir is None else toplevel_dir
    return RelativePath(toplevel_dir, os.path.join(toplevel_dir, path))