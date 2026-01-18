import sys
import os
import struct
import logging
import numpy as np
def _read_data_elements(self):
    info = self._info
    try:
        while True:
            group, element, value = self._readDataElement()
            if group in GROUPS:
                key = (group, element)
                name, vr = MINIDICT.get(key, (None, None))
                if name:
                    converter = self._converters.get(vr, lambda x: x)
                    info[name] = converter(value)
    except (EOFError, struct.error):
        pass