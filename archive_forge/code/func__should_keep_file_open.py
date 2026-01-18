from __future__ import annotations
import typing as ty
import warnings
from contextlib import contextmanager
from threading import RLock
import numpy as np
from . import openers
from .fileslice import canonical_slicers, fileslice
from .volumeutils import apply_read_scaling, array_from_file
def _should_keep_file_open(self, keep_file_open):
    """Called by ``__init__``.

        This method determines how to manage ``ImageOpener`` instances,
        and the underlying file handles - the behaviour depends on:

         - whether ``self.file_like`` is an an open file handle, or a path to a
           ``'.gz'`` file, or a path to a non-gzip file.
         - whether ``indexed_gzip`` is present (see
           :attr:`.openers.HAVE_INDEXED_GZIP`).

        An ``ArrayProxy`` object uses two internal flags to manage
        ``ImageOpener`` instances and underlying file handles.

          - The ``_persist_opener`` flag controls whether a single
            ``ImageOpener`` should be created and used for the lifetime of
            this ``ArrayProxy``, or whether separate ``ImageOpener`` instances
            should be created on each file access.

          - The ``_keep_file_open`` flag controls qwhether the underlying file
            handle should be kept open for the lifetime of this
            ``ArrayProxy``, or whether the file handle should be (re-)opened
            and closed on each file access.

        The internal ``_keep_file_open`` flag is only relevant if
        ``self.file_like`` is a ``'.gz'`` file, and the ``indexed_gzip`` library is
        present.

        This method returns the values to be used for the internal
        ``_persist_opener`` and ``_keep_file_open`` flags; these values are
        derived according to the following rules:

        1. If ``self.file_like`` is a file(-like) object, both flags are set to
        ``False``.

        2. If ``keep_file_open`` (as passed to :meth:``__init__``) is
           ``True``, both internal flags are set to ``True``.

        3. If ``keep_file_open`` is ``False``, but ``self.file_like`` is not a path
           to a ``.gz`` file or ``indexed_gzip`` is not present, both flags
           are set to ``False``.

        4. If ``keep_file_open`` is ``False``, ``self.file_like`` is a path to a
           ``.gz`` file, and ``indexed_gzip`` is present, ``_persist_opener``
           is set to ``True``, and ``_keep_file_open`` is set to ``False``.
           In this case, file handle management is delegated to the
           ``indexed_gzip`` library.

        Parameters
        ----------

        keep_file_open : { True, False }
            Flag as passed to ``__init__``.

        Returns
        -------

        A tuple containing:
          - ``keep_file_open`` flag to control persistence of file handles
          - ``persist_opener`` flag to control persistence of ``ImageOpener``
            objects.
        """
    if keep_file_open is None:
        keep_file_open = KEEP_FILE_OPEN_DEFAULT
        if keep_file_open not in (True, False):
            raise ValueError(f'nibabel.arrayproxy.KEEP_FILE_OPEN_DEFAULT must be boolean. Found: {keep_file_open}')
    elif keep_file_open not in (True, False):
        raise ValueError('keep_file_open must be one of {None, True, False}')
    if self._has_fh():
        return (False, False)
    have_igzip = openers.HAVE_INDEXED_GZIP and self.file_like.endswith('.gz')
    persist_opener = keep_file_open or have_igzip
    return (keep_file_open, persist_opener)