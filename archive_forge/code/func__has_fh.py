from __future__ import annotations
import typing as ty
import warnings
from contextlib import contextmanager
from threading import RLock
import numpy as np
from . import openers
from .fileslice import canonical_slicers, fileslice
from .volumeutils import apply_read_scaling, array_from_file
def _has_fh(self) -> bool:
    """Determine if our file-like is a filehandle or path"""
    return hasattr(self.file_like, 'read') and hasattr(self.file_like, 'seek')