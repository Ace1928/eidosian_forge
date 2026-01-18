import os.path
import struct
import zipfile
import zlib
class DeflatedZipFileEntry(_FileEntry):
    """
    File-like object used to read a deflated entry in a ZipFile
    """

    def __init__(self, chunkingZipFile, length):
        _FileEntry.__init__(self, chunkingZipFile, length)
        self.returnedBytes = 0
        self.readBytes = 0
        self.decomp = zlib.decompressobj(-15)
        self.buffer = b''

    def tell(self):
        return self.returnedBytes

    def read(self, n=None):
        if self.finished:
            return b''
        if n is None:
            result = [self.buffer]
            result.append(self.decomp.decompress(self.chunkingZipFile.fp.read(self.length - self.readBytes)))
            result.append(self.decomp.decompress(b'Z'))
            result.append(self.decomp.flush())
            self.buffer = b''
            self.finished = 1
            result = b''.join(result)
            self.returnedBytes += len(result)
            return result
        else:
            while len(self.buffer) < n:
                data = self.chunkingZipFile.fp.read(min(n, 1024, self.length - self.readBytes))
                self.readBytes += len(data)
                if not data:
                    result = self.buffer + self.decomp.decompress(b'Z') + self.decomp.flush()
                    self.finished = 1
                    self.buffer = b''
                    self.returnedBytes += len(result)
                    return result
                else:
                    self.buffer += self.decomp.decompress(data)
            result = self.buffer[:n]
            self.buffer = self.buffer[n:]
            self.returnedBytes += len(result)
            return result