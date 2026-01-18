from struct import pack, unpack, calcsize, error, Struct
import os
import sys
import time
import array
import tempfile
import logging
import io
from datetime import date
import zipfile
def __restrictIndex(self, i):
    """Provides list-like handling of a record index with a clearer
        error message if the index is out of bounds."""
    if self.numRecords:
        rmax = self.numRecords - 1
        if abs(i) > rmax:
            raise IndexError('Shape or Record index out of range.')
        if i < 0:
            i = range(self.numRecords)[i]
    return i