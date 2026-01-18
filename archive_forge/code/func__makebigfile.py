import random
import struct
import zipfile
from hashlib import md5
from twisted.python import filepath, zipstream
from twisted.trial import unittest
def _makebigfile(self, filename, compression, junk):
    """
        Create a zip file with the given file name and compression scheme.
        """
    with zipfile.ZipFile(filename, 'w', compression) as zf:
        for i in range(10):
            fn = 'zipstream%d' % i
            zf.writestr(fn, '')
        zf.writestr('zipstreamjunk', junk)