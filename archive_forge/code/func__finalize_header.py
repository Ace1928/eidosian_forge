import os
import warnings
from contextlib import suppress
import numpy as np
from nibabel.openers import Opener
from .array_sequence import ArraySequence
from .header import Field
from .tractogram import LazyTractogram, Tractogram, TractogramItem
from .tractogram_file import DataError, DataWarning, HeaderError, HeaderWarning, TractogramFile
from .utils import peek_next
def _finalize_header(self, f, header, offset=0):
    f.seek(offset, os.SEEK_SET)
    self._write_header(f, header)