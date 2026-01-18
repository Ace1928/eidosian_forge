import random
import struct
import zipfile
from hashlib import md5
from twisted.python import filepath, zipstream
from twisted.trial import unittest
def _unzipIterChunkyTest(self, compression, chunksize, lower, upper):
    """
        unzipIterChunky should unzip the given number of bytes per iteration.
        """
    junk = b''
    for n in range(1000):
        num = round(random.random(), 12)
        numEncoded = str(num).encode('ascii')
        junk += b' ' + numEncoded
    junkmd5 = md5(junk).hexdigest()
    tempdir = filepath.FilePath(self.mktemp())
    tempdir.makedirs()
    zfpath = tempdir.child('bigfile.zip').path
    self._makebigfile(zfpath, compression, junk)
    uziter = zipstream.unzipIterChunky(zfpath, tempdir.path, chunksize=chunksize)
    r = next(uziter)
    approx = lower < r < upper
    self.assertTrue(approx)
    for r in uziter:
        pass
    self.assertEqual(r, 0)
    with tempdir.child('zipstreamjunk').open() as f:
        newmd5 = md5(f.read()).hexdigest()
        self.assertEqual(newmd5, junkmd5)