from array import array
from bisect import bisect_right
from os.path import isfile
from struct import Struct
from warnings import warn
from pyzstd.zstdfile import ZstdDecompressReader, ZstdFile, \
class SeekableDecompressReader(ZstdDecompressReader):

    def __init__(self, fp, zstd_dict, option, read_size):
        if not hasattr(fp, 'readable') or not hasattr(fp, 'seekable'):
            raise TypeError("In SeekableZstdFile's reading mode, the file object should have .readable()/.seekable() methods.")
        if not fp.readable():
            raise TypeError("In SeekableZstdFile's reading mode, the file object should be readable.")
        if not fp.seekable():
            raise TypeError("In SeekableZstdFile's reading mode, the file object should be seekable. If the file object is not seekable, it can be read sequentially using ZstdFile class.")
        self._seek_table = SeekTable(read_mode=True)
        self._seek_table.load_seek_table(fp, seek_to_0=True)
        super().__init__(fp, zstd_dict, option, read_size)
        self._decomp.size = self._seek_table.get_full_d_size()

    def seekable(self):
        return True

    def seek(self, offset, whence=0):
        if whence == 0:
            pass
        elif whence == 1:
            offset = self._decomp.pos + offset
        elif whence == 2:
            offset = self._decomp.size + offset
        else:
            raise ValueError('Invalid value for whence: {}'.format(whence))
        new_frame = self._seek_table.index_by_dpos(offset)
        if new_frame is None:
            self._decomp.eof = True
            self._decomp.pos = self._decomp.size
            self._fp.seek(self._seek_table.file_size)
            return self._decomp.pos
        old_frame = self._seek_table.index_by_dpos(self._decomp.pos)
        c_pos, d_pos = self._seek_table.get_frame_sizes(new_frame)
        if new_frame == old_frame and offset >= self._decomp.pos and (self._fp.tell() >= c_pos):
            pass
        else:
            self._decomp.eof = False
            self._decomp.pos = d_pos
            self._decomp.reset_session()
            self._fp.seek(c_pos)
        offset -= self._decomp.pos
        self._decomp.forward(offset)
        return self._decomp.pos

    def get_seek_table_info(self):
        return self._seek_table.get_info()