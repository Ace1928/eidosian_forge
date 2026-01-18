import os
import json
import struct
import logging
import numpy as np
from ..core import Format
from ..v2 import imread
class LytroFormat(Format):
    """Base class for Lytro format.
    The subclasses LytroLfrFormat, LytroLfpFormat, LytroIllumRawFormat and
    LytroF01RawFormat implement the Lytro-LFR, Lytro-LFP and Lytro-RAW format
    for the Illum and original F01 camera respectively.
    Writing is not supported.
    """
    _modes = 'i'

    def _can_write(self, request):
        return False

    class Writer(Format.Writer):

        def _open(self, flags=0):
            self._fp = self.request.get_file()

        def _close(self):
            pass

        def _append_data(self, im, meta):
            raise RuntimeError('The lytro format cannot write image data.')

        def _set_meta_data(self, meta):
            raise RuntimeError('The lytro format cannot write meta data.')