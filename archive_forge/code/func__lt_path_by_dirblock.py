import binascii
import os
import struct
from .dirstate import DirState, DirstateCorrupt
def _lt_path_by_dirblock(path1, path2):
    """Compare two paths based on what directory they are in.

    This generates a sort order, such that all children of a directory are
    sorted together, and grandchildren are in the same order as the
    children appear. But all grandchildren come after all children.

    :param path1: first path
    :param path2: the second path
    :return: True if path1 comes first, otherwise False
    """
    if not isinstance(path1, bytes):
        raise TypeError("'path1' must be a plain string, not %s: %r" % (type(path1), path1))
    if not isinstance(path2, bytes):
        raise TypeError("'path2' must be a plain string, not %s: %r" % (type(path2), path2))
    dirname1, basename1 = os.path.split(path1)
    key1 = (dirname1.split(b'/'), basename1)
    dirname2, basename2 = os.path.split(path2)
    key2 = (dirname2.split(b'/'), basename2)
    return key1 < key2