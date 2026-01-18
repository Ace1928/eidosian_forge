import os
import contextlib
import itertools
import collections.abc
from abc import ABC, abstractmethod
def _open_for_random_access(filename):
    """Open a file in binary mode, spot if it is BGZF format etc (PRIVATE).

    This functionality is used by the Bio.SeqIO and Bio.SearchIO index
    and index_db functions.

    If the file is gzipped but not BGZF, a specific ValueError is raised.
    """
    handle = open(filename, 'rb')
    magic = handle.read(2)
    handle.seek(0)
    if magic == b'\x1f\x8b':
        from . import bgzf
        try:
            return bgzf.BgzfReader(mode='rb', fileobj=handle)
        except ValueError as e:
            assert 'BGZF' in str(e)
            handle.close()
            raise ValueError('Gzipped files are not suitable for indexing, please use BGZF (blocked gzip format) instead.') from None
    return handle