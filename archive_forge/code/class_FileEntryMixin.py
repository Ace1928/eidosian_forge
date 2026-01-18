import random
import struct
import zipfile
from hashlib import md5
from twisted.python import filepath, zipstream
from twisted.trial import unittest
class FileEntryMixin:
    """
    File entry classes should behave as file-like objects
    """

    def getFileEntry(self, contents):
        """
        Return an appropriate zip file entry
        """
        filename = self.mktemp()
        with zipfile.ZipFile(filename, 'w', self.compression) as z:
            z.writestr('content', contents)
        z = zipstream.ChunkingZipFile(filename, 'r')
        return z.readfile('content')

    def test_isatty(self):
        """
        zip files should not be ttys, so isatty() should be false
        """
        with self.getFileEntry('') as fileEntry:
            self.assertFalse(fileEntry.isatty())

    def test_closed(self):
        """
        The C{closed} attribute should reflect whether C{close()} has been
        called.
        """
        with self.getFileEntry('') as fileEntry:
            self.assertFalse(fileEntry.closed)
        self.assertTrue(fileEntry.closed)

    def test_readline(self):
        """
        C{readline()} should mirror L{file.readline} and return up to a single
        delimiter.
        """
        with self.getFileEntry(b'hoho\nho') as fileEntry:
            self.assertEqual(fileEntry.readline(), b'hoho\n')
            self.assertEqual(fileEntry.readline(), b'ho')
            self.assertEqual(fileEntry.readline(), b'')

    def test_next(self):
        """
        Zip file entries should implement the iterator protocol as files do.
        """
        with self.getFileEntry(b'ho\nhoho') as fileEntry:
            self.assertEqual(fileEntry.next(), b'ho\n')
            self.assertEqual(fileEntry.next(), b'hoho')
            self.assertRaises(StopIteration, fileEntry.next)

    def test_readlines(self):
        """
        C{readlines()} should return a list of all the lines.
        """
        with self.getFileEntry(b'ho\nho\nho') as fileEntry:
            self.assertEqual(fileEntry.readlines(), [b'ho\n', b'ho\n', b'ho'])

    def test_iteration(self):
        """
        C{__iter__()} and C{xreadlines()} should return C{self}.
        """
        with self.getFileEntry('') as fileEntry:
            self.assertIs(iter(fileEntry), fileEntry)
            self.assertIs(fileEntry.xreadlines(), fileEntry)

    def test_readWhole(self):
        """
        C{.read()} should read the entire file.
        """
        contents = b'Hello, world!'
        with self.getFileEntry(contents) as entry:
            self.assertEqual(entry.read(), contents)

    def test_readPartial(self):
        """
        C{.read(num)} should read num bytes from the file.
        """
        contents = '0123456789'
        with self.getFileEntry(contents) as entry:
            one = entry.read(4)
            two = entry.read(200)
        self.assertEqual(one, b'0123')
        self.assertEqual(two, b'456789')

    def test_tell(self):
        """
        C{.tell()} should return the number of bytes that have been read so
        far.
        """
        contents = 'x' * 100
        with self.getFileEntry(contents) as entry:
            entry.read(2)
            self.assertEqual(entry.tell(), 2)
            entry.read(4)
            self.assertEqual(entry.tell(), 6)