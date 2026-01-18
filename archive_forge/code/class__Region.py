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
class _Region:
    __slots__ = ('chromId', 'start', 'end', 'offset')

    def __init__(self, chromId, start, end):
        self.chromId = chromId
        self.start = start
        self.end = end