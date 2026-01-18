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
class _ZoomLevel:
    __slots__ = ['amount', 'dataOffset', 'indexOffset']
    formatter = struct.Struct('=IxxxxQQ')

    def __bytes__(self):
        return self.formatter.pack(self.amount, self.dataOffset, self.indexOffset)