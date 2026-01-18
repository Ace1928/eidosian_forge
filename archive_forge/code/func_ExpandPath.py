from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import fnmatch
import os
import re
from googlecloudsdk.api_lib.storage import storage_api
from googlecloudsdk.api_lib.storage import storage_util
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
import six
def ExpandPath(self, path):
    """Expand the given path that contains wildcard characters.

    Args:
      path: str, The path to expand.

    Returns:
      ({str}, {str}), A tuple of the sets of files and directories that match
      the wildcard path. All returned paths are absolute.
    """
    files = set()
    dirs = set()
    for p in self._Glob(self.AbsPath(path)):
        if p.endswith(self._sep):
            dirs.add(p)
        else:
            files.add(p)
    if self.IsEndRecursive(path):
        dirs.clear()
    return (files, dirs)