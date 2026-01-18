import os
import abc
import codecs
import errno
import stat
import sys
from _thread import allocate_lock as Lock
import io
from io import (__all__, SEEK_SET, SEEK_CUR, SEEK_END)
from _io import FileIO
def _reset_encoder(position):
    """Reset the encoder (merely useful for proper BOM handling)"""
    try:
        encoder = self._encoder or self._get_encoder()
    except LookupError:
        pass
    else:
        if position != 0:
            encoder.setstate(0)
        else:
            encoder.reset()