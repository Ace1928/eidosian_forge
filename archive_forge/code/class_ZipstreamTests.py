import random
import struct
import zipfile
from hashlib import md5
from twisted.python import filepath, zipstream
from twisted.trial import unittest
class ZipstreamTests(unittest.TestCase):
    """
    Tests for twisted.python.zipstream
    """

    def setUp(self):
        """
        Creates junk data that can be compressed and a test directory for any
        files that will be created
        """
        self.testdir = filepath.FilePath(self.mktemp())
        self.testdir.makedirs()
        self.unzipdir = self.testdir.child('unzipped')
        self.unzipdir.makedirs()

    def makeZipFile(self, contents, directory=''):
        """
        Makes a zip file archive containing len(contents) files.  Contents
        should be a list of strings, each string being the content of one file.
        """
        zpfilename = self.testdir.child('zipfile.zip').path
        with zipfile.ZipFile(zpfilename, 'w') as zpfile:
            for i, content in enumerate(contents):
                filename = str(i)
                if directory:
                    filename = directory + '/' + filename
                zpfile.writestr(filename, content)
        return zpfilename

    def test_invalidMode(self):
        """
        A ChunkingZipFile opened in write-mode should not allow .readfile(),
        and raise a RuntimeError instead.
        """
        with zipstream.ChunkingZipFile(self.mktemp(), 'w') as czf:
            self.assertRaises(RuntimeError, czf.readfile, 'something')

    def test_closedArchive(self):
        """
        A closed ChunkingZipFile should raise a L{RuntimeError} when
        .readfile() is invoked.
        """
        czf = zipstream.ChunkingZipFile(self.makeZipFile(['something']), 'r')
        czf.close()
        self.assertRaises(RuntimeError, czf.readfile, 'something')

    def test_invalidHeader(self):
        """
        A zipfile entry with the wrong magic number should raise BadZipFile for
        readfile(), but that should not affect other files in the archive.
        """
        fn = self.makeZipFile(['test contents', 'more contents'])
        with zipfile.ZipFile(fn, 'r') as zf:
            zeroOffset = zf.getinfo('0').header_offset
        with open(fn, 'r+b') as scribble:
            scribble.seek(zeroOffset, 0)
            scribble.write(b'0' * 4)
        with zipstream.ChunkingZipFile(fn) as czf:
            self.assertRaises(zipfile.BadZipFile, czf.readfile, '0')
            with czf.readfile('1') as zfe:
                self.assertEqual(zfe.read(), b'more contents')

    def test_filenameMismatch(self):
        """
        A zipfile entry with a different filename than is found in the central
        directory should raise BadZipFile.
        """
        fn = self.makeZipFile([b'test contents', b'more contents'])
        with zipfile.ZipFile(fn, 'r') as zf:
            info = zf.getinfo('0')
            info.filename = 'not zero'
        with open(fn, 'r+b') as scribble:
            scribble.seek(info.header_offset, 0)
            scribble.write(info.FileHeader())
        with zipstream.ChunkingZipFile(fn) as czf:
            self.assertRaises(zipfile.BadZipFile, czf.readfile, '0')
            with czf.readfile('1') as zfe:
                self.assertEqual(zfe.read(), b'more contents')

    def test_unsupportedCompression(self):
        """
        A zipfile which describes an unsupported compression mechanism should
        raise BadZipFile.
        """
        fn = self.mktemp()
        with zipfile.ZipFile(fn, 'w') as zf:
            zi = zipfile.ZipInfo('0')
            zf.writestr(zi, 'some data')
            zi.compress_type = 1234
        with zipstream.ChunkingZipFile(fn) as czf:
            self.assertRaises(zipfile.BadZipFile, czf.readfile, '0')

    def test_extraData(self):
        """
        readfile() should skip over 'extra' data present in the zip metadata.
        """
        fn = self.mktemp()
        with zipfile.ZipFile(fn, 'w') as zf:
            zi = zipfile.ZipInfo('0')
            extra_data = b'hello, extra'
            zi.extra = struct.pack('<hh', 42, len(extra_data)) + extra_data
            zf.writestr(zi, b'the real data')
        with zipstream.ChunkingZipFile(fn) as czf, czf.readfile('0') as zfe:
            self.assertEqual(zfe.read(), b'the real data')

    def test_unzipIterChunky(self):
        """
        L{twisted.python.zipstream.unzipIterChunky} returns an iterator which
        must be exhausted to completely unzip the input archive.
        """
        numfiles = 10
        contents = ['This is test file %d!' % i for i in range(numfiles)]
        contents = [i.encode('ascii') for i in contents]
        zpfilename = self.makeZipFile(contents)
        list(zipstream.unzipIterChunky(zpfilename, self.unzipdir.path))
        self.assertEqual(set(self.unzipdir.listdir()), set(map(str, range(numfiles))))
        for child in self.unzipdir.children():
            num = int(child.basename())
            self.assertEqual(child.getContent(), contents[num])

    def test_unzipIterChunkyDirectory(self):
        """
        The path to which a file is extracted by L{zipstream.unzipIterChunky}
        is determined by joining the C{directory} argument to C{unzip} with the
        path within the archive of the file being extracted.
        """
        numfiles = 10
        contents = ['This is test file %d!' % i for i in range(numfiles)]
        contents = [i.encode('ascii') for i in contents]
        zpfilename = self.makeZipFile(contents, 'foo')
        list(zipstream.unzipIterChunky(zpfilename, self.unzipdir.path))
        fileContents = {str(num).encode('ascii') for num in range(numfiles)}
        self.assertEqual(set(self.unzipdir.child(b'foo').listdir()), fileContents)
        for child in self.unzipdir.child(b'foo').children():
            num = int(child.basename())
            self.assertEqual(child.getContent(), contents[num])

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

    def test_unzipIterChunkyStored(self):
        """
        unzipIterChunky should unzip the given number of bytes per iteration on
        a stored archive.
        """
        self._unzipIterChunkyTest(zipfile.ZIP_STORED, 500, 35, 45)

    def test_chunkyDeflated(self):
        """
        unzipIterChunky should unzip the given number of bytes per iteration on
        a deflated archive.
        """
        self._unzipIterChunkyTest(zipfile.ZIP_DEFLATED, 972, 23, 27)

    def _makebigfile(self, filename, compression, junk):
        """
        Create a zip file with the given file name and compression scheme.
        """
        with zipfile.ZipFile(filename, 'w', compression) as zf:
            for i in range(10):
                fn = 'zipstream%d' % i
                zf.writestr(fn, '')
            zf.writestr('zipstreamjunk', junk)