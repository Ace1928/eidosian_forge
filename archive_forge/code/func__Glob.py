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
def _Glob(self, path):
    if not self.HasExpansion(path):
        if self.Exists(path):
            yield self._FormatPath(path)
        return
    dir_path, basename = os.path.split(path)
    has_basename_expansion = self.HasExpansion(basename)
    for expanded_dir_path in self._Glob(dir_path):
        if not has_basename_expansion:
            path = self.Join(expanded_dir_path, basename)
            if self.Exists(path):
                yield self._FormatPath(path)
        elif basename == '**':
            for n in self._RecursiveDirList(expanded_dir_path):
                yield self._FormatPath(n)
        else:
            for n in fnmatch.filter(self.ListDir(expanded_dir_path), basename):
                yield self._FormatPath(self.Join(expanded_dir_path, n))