import posixpath
import stat
import struct
import tarfile
from contextlib import closing
from io import BytesIO
from os import SEEK_END
def _walk_tree(store, tree, root=b''):
    """Recursively walk a dulwich Tree, yielding tuples of
    (absolute path, TreeEntry) along the way.
    """
    for entry in tree.iteritems():
        entry_abspath = posixpath.join(root, entry.path)
        if stat.S_ISDIR(entry.mode):
            yield from _walk_tree(store, store[entry.sha], entry_abspath)
        else:
            yield (entry_abspath, entry)