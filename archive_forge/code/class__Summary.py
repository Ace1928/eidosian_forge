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
class _Summary:
    __slots__ = ('validCount', 'minVal', 'maxVal', 'sumData', 'sumSquares')
    formatter = struct.Struct('=Qdddd')
    size = formatter.size

    def __init__(self):
        self.validCount = 0
        self.minVal = sys.maxsize
        self.maxVal = -sys.maxsize
        self.sumData = 0.0
        self.sumSquares = 0.0

    def update(self, size, val):
        self.validCount += size
        if val < self.minVal:
            self.minVal = val
        if val > self.maxVal:
            self.maxVal = val
        self.sumData += val * size
        self.sumSquares += val * val * size

    def __bytes__(self):
        return self.formatter.pack(self.validCount, self.minVal, self.maxVal, self.sumData, self.sumSquares)