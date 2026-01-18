import dataclasses
import fnmatch
import logging
import os
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Tuple, Type, Union
from ray._private.storage import _get_storage_uri
from ray.air._internal.filelock import TempFileLock
from ray.train._internal.syncer import SyncConfig, Syncer, _BackgroundSyncer
from ray.train.constants import _get_defaults_results_dir
class _ExcludingLocalFilesystem(LocalFileSystem):
    """LocalFileSystem wrapper to exclude files according to patterns.

    Args:
        root_path: Root path to strip when matching with the exclude pattern.
            Ex: root_path="/tmp/a/b/c", exclude=["*a*"], will exclude
            /tmp/a/b/c/_a_.txt but not ALL of /tmp/a/*.
        exclude: List of patterns that are applied to files returned by
            ``self.find()``. If a file path matches this pattern, it will
            be excluded.

    """

    def __init__(self, root_path: Path, exclude: List[str], **kwargs):
        super().__init__(**kwargs)
        self._exclude = exclude
        self._root_path = root_path

    @property
    def fsid(self):
        return '_excluding_local'

    def _should_exclude(self, path: str) -> bool:
        """Return True if `path` (relative to `root_path`) matches any of the
        `self._exclude` patterns."""
        path = Path(path)
        relative_path = path.relative_to(self._root_path).as_posix()
        alt = os.path.join(relative_path, '') if path.is_dir() else None
        for excl in self._exclude:
            if fnmatch.fnmatch(relative_path, excl):
                return True
            if alt and fnmatch.fnmatch(alt, excl):
                return True
        return False

    def find(self, path, maxdepth=None, withdirs=False, detail=False, **kwargs):
        """Call parent find() and exclude from result."""
        paths = super().find(path, maxdepth=maxdepth, withdirs=withdirs, detail=detail, **kwargs)
        if detail:
            return {path: out for path, out in paths.items() if not self._should_exclude(path)}
        else:
            return [path for path in paths if not self._should_exclude(path)]