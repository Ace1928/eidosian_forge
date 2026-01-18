import posixpath
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Callable, Dict, List, Optional
from ray.util.annotations import DeveloperAPI, PublicAPI
def _normalize_base_dir(self):
    """Normalizes the partition base directory for compatibility with the
        given filesystem.

        This should be called once a filesystem has been resolved to ensure that this
        base directory is correctly discovered at the root of all partitioned file
        paths.
        """
    from ray.data.datasource.path_util import _resolve_paths_and_filesystem
    paths, self._resolved_filesystem = _resolve_paths_and_filesystem(self.base_dir, self.filesystem)
    assert len(paths) == 1, f'Expected 1 normalized base directory, but found {len(paths)}'
    normalized_base_dir = paths[0]
    if len(normalized_base_dir) and (not normalized_base_dir.endswith('/')):
        normalized_base_dir += '/'
    self._normalized_base_dir = normalized_base_dir