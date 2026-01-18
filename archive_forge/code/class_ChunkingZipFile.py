import os.path
import struct
import zipfile
import zlib
class ChunkingZipFile(zipfile.ZipFile):
    """
    A L{zipfile.ZipFile} object which, with L{readfile}, also gives you access
    to a file-like object for each entry.
    """

    def readfile(self, name):
        """
        Return file-like object for name.
        """
        if self.mode not in ('r', 'a'):
            raise RuntimeError('read() requires mode "r" or "a"')
        if not self.fp:
            raise RuntimeError('Attempt to read ZIP archive that was already closed')
        zinfo = self.getinfo(name)
        self.fp.seek(zinfo.header_offset, 0)
        fheader = self.fp.read(zipfile.sizeFileHeader)
        if fheader[0:4] != zipfile.stringFileHeader:
            raise zipfile.BadZipFile('Bad magic number for file header')
        fheader = struct.unpack(zipfile.structFileHeader, fheader)
        fname = self.fp.read(fheader[zipfile._FH_FILENAME_LENGTH])
        if fheader[zipfile._FH_EXTRA_FIELD_LENGTH]:
            self.fp.read(fheader[zipfile._FH_EXTRA_FIELD_LENGTH])
        if zinfo.flag_bits & 2048:
            fname_str = fname.decode('utf-8')
        else:
            fname_str = fname.decode('cp437')
        if fname_str != zinfo.orig_filename:
            raise zipfile.BadZipFile('File name in directory "%s" and header "%s" differ.' % (zinfo.orig_filename, fname_str))
        if zinfo.compress_type == zipfile.ZIP_STORED:
            return ZipFileEntry(self, zinfo.compress_size)
        elif zinfo.compress_type == zipfile.ZIP_DEFLATED:
            return DeflatedZipFileEntry(self, zinfo.compress_size)
        else:
            raise zipfile.BadZipFile('Unsupported compression method %d for file %s' % (zinfo.compress_type, name))