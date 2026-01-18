import binascii
import os
import struct
from .dirstate import DirState, DirstateCorrupt
def _bisect_path_left(paths, path):
    """Return the index where to insert path into paths.

    This uses the dirblock sorting. So all children in a directory come before
    the children of children. For example::

        a/
          b/
            c
          d/
            e
          b-c
          d-e
        a-a
        a=c

    Will be sorted as::

        a
        a-a
        a=c
        a/b
        a/b-c
        a/d
        a/d-e
        a/b/c
        a/d/e

    :param paths: A list of paths to search through
    :param path: A single path to insert
    :return: An offset where 'path' can be inserted.
    :seealso: bisect.bisect_left
    """
    hi = len(paths)
    lo = 0
    while lo < hi:
        mid = (lo + hi) // 2
        cur = paths[mid]
        if _lt_path_by_dirblock(cur, path):
            lo = mid + 1
        else:
            hi = mid
    return lo