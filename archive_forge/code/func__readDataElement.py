import sys
import os
import struct
import logging
import numpy as np
def _readDataElement(self):
    f = self._file
    group = self._unpack('H', f.read(2))
    element = self._unpack('H', f.read(2))
    if self.is_implicit_VR:
        vl = self._unpack('I', f.read(4))
    else:
        vr = f.read(2)
        if vr in (b'OB', b'OW', b'SQ', b'UN'):
            reserved = f.read(2)
            vl = self._unpack('I', f.read(4))
        else:
            vl = self._unpack('H', f.read(2))
    if group == 32736 and element == 16:
        here = f.tell()
        self._pixel_data_loc = (here, vl)
        f.seek(here + vl)
        return (group, element, b'Deferred loading of pixel data')
    else:
        if vl == 4294967295:
            value = self._read_undefined_length_value()
        else:
            value = f.read(vl)
        return (group, element, value)