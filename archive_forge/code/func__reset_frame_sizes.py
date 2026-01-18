from array import array
from bisect import bisect_right
from os.path import isfile
from struct import Struct
from warnings import warn
from pyzstd.zstdfile import ZstdDecompressReader, ZstdFile, \
def _reset_frame_sizes(self):
    self._current_c_size = 0
    self._current_d_size = 0
    self._left_d_size = self._max_frame_content_size