from array import array
from bisect import bisect_right
from os.path import isfile
from struct import Struct
from warnings import warn
from pyzstd.zstdfile import ZstdDecompressReader, ZstdFile, \
def _merge_frames(self, max_frames):
    if self._frames_count <= max_frames:
        return
    arr = self._frames
    a, b = divmod(self._frames_count, max_frames)
    self._clear_seek_table()
    pos = 0
    for i in range(max_frames):
        length = (a + (1 if i < b else 0)) * 2
        c_size = 0
        d_size = 0
        for j in range(pos, pos + length, 2):
            c_size += arr[j]
            d_size += arr[j + 1]
        self.append_entry(c_size, d_size)
        pos += length