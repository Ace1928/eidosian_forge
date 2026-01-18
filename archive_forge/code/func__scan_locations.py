from __future__ import annotations
import os
import pickle
import time
from typing import TYPE_CHECKING
from fsspec.utils import atomic_write
def _scan_locations(self, writable_only: bool=False) -> Iterator[tuple[str, str, bool]]:
    """Yield locations (filenames) where metadata is stored, and whether
        writable or not.

        Parameters
        ----------
        writable: bool
            Set to True to only yield writable locations.

        Returns
        -------
        Yields (str, str, bool)
        """
    n = len(self._storage)
    for i, storage in enumerate(self._storage):
        writable = i == n - 1
        if writable_only and (not writable):
            continue
        yield (os.path.join(storage, 'cache'), storage, writable)