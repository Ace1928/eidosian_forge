import sys
import io
import copy
import array
import itertools
import struct
import zlib
from collections import namedtuple
from io import BytesIO
import numpy as np
from Bio.Align import Alignment
from Bio.Align import interfaces
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
class _Range:
    __slots__ = ('next', 'start', 'end', 'val')

    def __init__(self, start, end, val):
        self.start = start
        self.end = end
        self.val = val

    def __iter__(self):
        return iter((self.start, self.end, self.val))