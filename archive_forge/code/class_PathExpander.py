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
class PathExpander(six.with_metaclass(abc.ABCMeta)):
    """Abstract base class for path wildcard expansion."""
    EXPANSION_CHARS = '[*?[]'

    @classmethod
    def ForPath(cls, path):
        if path.startswith('gs://'):
            return GCSPathExpander()
        return LocalPathExpander()

    def __init__(self, sep):
        self._sep = sep

    @abc.abstractmethod
    def AbsPath(self, path):
        pass

    @abc.abstractmethod
    def IsFile(self, path):
        pass

    @abc.abstractmethod
    def IsDir(self, path):
        pass

    @abc.abstractmethod
    def Exists(self, path):
        pass

    @abc.abstractmethod
    def ListDir(self, path):
        pass

    @abc.abstractmethod
    def Join(self, path1, path2):
        pass

    @classmethod
    def HasExpansion(cls, path):
        return bool(re.search(PathExpander.EXPANSION_CHARS, path))

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

    def ExpandPaths(self, paths):
        files = set()
        dirs = set()
        for p in paths:
            current_files, current_dirs = self.ExpandPath(p)
            if not current_files and (not current_dirs):
                log.warning('[{}] does not match any paths.'.format(p))
                continue
            files.update(current_files)
            dirs.update(current_dirs)
        return (files, dirs)

    def IsEndRecursive(self, path):
        return path.endswith(self._sep + '**')

    def IsDirLike(self, path):
        return path.endswith(self._sep)

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

    def _RecursiveDirList(self, dir_path):
        for n in self.ListDir(dir_path):
            path = self.Join(dir_path, n)
            yield path
            for x in self._RecursiveDirList(path):
                yield x

    def _FormatPath(self, path):
        if self.IsDir(path) and (not path.endswith(self._sep)):
            path = path + self._sep
        return path