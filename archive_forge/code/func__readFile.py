import base64
import glob
import os
import pickle
from twisted.python.filepath import FilePath
def _readFile(self, path):
    """
        Read in the contents of a file.

        Override in subclasses to e.g. provide transparently encrypted dirdbm.
        """
    with _open(path.path, 'rb') as f:
        s = f.read()
    return s