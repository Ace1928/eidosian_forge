import io
import sys
from ctypes import *
import ctypes.util
import struct
from freetype.raw import *
def _init_name_string_map(self):
    self._name_strings = dict()
    for nidx in range(self._get_sfnt_name_count()):
        namerec = self.get_sfnt_name(nidx)
        nk = (namerec.name_id, namerec.platform_id, namerec.encoding_id, namerec.language_id)
        self._name_strings[nk] = namerec.string