from __future__ import annotations
import typing as ty
import warnings
from contextlib import contextmanager
from threading import RLock
import numpy as np
from . import openers
from .fileslice import canonical_slicers, fileslice
from .volumeutils import apply_read_scaling, array_from_file
@contextmanager
def _get_fileobj(self):
    """Create and return a new ``ImageOpener``, or return an existing one.

        The specific behaviour depends on the value of the ``keep_file_open``
        flag that was passed to ``__init__``.

        Yields
        ------
        ImageOpener
            A newly created ``ImageOpener`` instance, or an existing one,
            which provides access to the file.
        """
    if self._persist_opener:
        if not hasattr(self, '_opener'):
            self._opener = openers.ImageOpener(self.file_like, keep_open=self._keep_file_open)
        yield self._opener
    else:
        with openers.ImageOpener(self.file_like, keep_open=False) as opener:
            yield opener