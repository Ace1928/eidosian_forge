import sys
import os
import struct
import logging
import numpy as np
def _splitValues(self, x, type, splitter):
    s = x.decode('ascii').strip('\x00')
    try:
        if splitter in s:
            return tuple([type(v) for v in s.split(splitter) if v.strip()])
        else:
            return type(s)
    except ValueError:
        return s