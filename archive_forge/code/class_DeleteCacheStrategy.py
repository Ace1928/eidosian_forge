import os
import shutil
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, FrozenSet, List, Literal, Optional, Set, Union
from ..constants import HF_HUB_CACHE
from . import logging
@dataclass(frozen=True)
class DeleteCacheStrategy:
    """Frozen data structure holding the strategy to delete cached revisions.

    This object is not meant to be instantiated programmatically but to be returned by
    [`~utils.HFCacheInfo.delete_revisions`]. See documentation for usage example.

    Args:
        expected_freed_size (`float`):
            Expected freed size once strategy is executed.
        blobs (`FrozenSet[Path]`):
            Set of blob file paths to be deleted.
        refs (`FrozenSet[Path]`):
            Set of reference file paths to be deleted.
        repos (`FrozenSet[Path]`):
            Set of entire repo paths to be deleted.
        snapshots (`FrozenSet[Path]`):
            Set of snapshots to be deleted (directory of symlinks).
    """
    expected_freed_size: int
    blobs: FrozenSet[Path]
    refs: FrozenSet[Path]
    repos: FrozenSet[Path]
    snapshots: FrozenSet[Path]

    @property
    def expected_freed_size_str(self) -> str:
        """
        (property) Expected size that will be freed as a human-readable string.

        Example: "42.2K".
        """
        return _format_size(self.expected_freed_size)

    def execute(self) -> None:
        """Execute the defined strategy.

        <Tip warning={true}>

        If this method is interrupted, the cache might get corrupted. Deletion order is
        implemented so that references and symlinks are deleted before the actual blob
        files.

        </Tip>

        <Tip warning={true}>

        This method is irreversible. If executed, cached files are erased and must be
        downloaded again.

        </Tip>
        """
        for path in self.repos:
            _try_delete_path(path, path_type='repo')
        for path in self.snapshots:
            _try_delete_path(path, path_type='snapshot')
        for path in self.refs:
            _try_delete_path(path, path_type='ref')
        for path in self.blobs:
            _try_delete_path(path, path_type='blob')
        logger.info(f'Cache deletion done. Saved {self.expected_freed_size_str}.')