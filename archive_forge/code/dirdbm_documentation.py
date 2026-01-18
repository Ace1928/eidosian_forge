import base64
import glob
import os
import pickle
from twisted.python.filepath import FilePath

        C{dirdbm[foo]}
        Get and unpickle the contents of a file in this directory.

        @type k: bytes
        @param k: The key to lookup

        @return: The value associated with the given key
        @raise KeyError: Raised if the given key does not exist
        