import os.path
import struct
import zipfile
import zlib
class _FileEntry:
    """
    Abstract superclass of both compressed and uncompressed variants of
    file-like objects within a zip archive.

    @ivar chunkingZipFile: a chunking zip file.
    @type chunkingZipFile: L{ChunkingZipFile}

    @ivar length: The number of bytes within the zip file that represent this
    file.  (This is the size on disk, not the number of decompressed bytes
    which will result from reading it.)

    @ivar fp: the underlying file object (that contains pkzip data).  Do not
    touch this, please.  It will quite likely move or go away.

    @ivar closed: File-like 'closed' attribute; True before this file has been
    closed, False after.
    @type closed: L{bool}

    @ivar finished: An older, broken synonym for 'closed'.  Do not touch this,
    please.
    @type finished: L{int}
    """

    def __init__(self, chunkingZipFile, length):
        """
        Create a L{_FileEntry} from a L{ChunkingZipFile}.
        """
        self.chunkingZipFile = chunkingZipFile
        self.fp = self.chunkingZipFile.fp
        self.length = length
        self.finished = 0
        self.closed = False

    def isatty(self):
        """
        Returns false because zip files should not be ttys
        """
        return False

    def close(self):
        """
        Close self (file-like object)
        """
        self.closed = True
        self.finished = 1
        del self.fp

    def readline(self):
        """
        Read a line.
        """
        line = b''
        for byte in iter(lambda: self.read(1), b''):
            line += byte
            if byte == b'\n':
                break
        return line

    def __next__(self):
        """
        Implement next as file does (like readline, except raises StopIteration
        at EOF)
        """
        nextline = self.readline()
        if nextline:
            return nextline
        raise StopIteration()
    next = __next__

    def readlines(self):
        """
        Returns a list of all the lines
        """
        return list(self)

    def xreadlines(self):
        """
        Returns an iterator (so self)
        """
        return self

    def __iter__(self):
        """
        Returns an iterator (so self)
        """
        return self

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()