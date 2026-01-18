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
def _iterate_index(self, stream):
    formatter = struct.Struct(self.byteorder + 'III')
    size = formatter.size
    node = self.tree
    while True:
        try:
            children = node.children
        except AttributeError:
            stream.seek(node.dataOffset)
            data = stream.read(node.dataSize)
            if self._compressed > 0:
                data = zlib.decompress(data)
            while data:
                chromId, chromStart, chromEnd = formatter.unpack(data[:size])
                rest, data = data[size:].split(b'\x00', 1)
                yield (chromId, chromStart, chromEnd, rest)
            while True:
                parent = node.parent
                if parent is None:
                    return
                for index, child in enumerate(parent.children):
                    if id(node) == id(child):
                        break
                else:
                    raise RuntimeError('Failed to find child node')
                try:
                    node = parent.children[index + 1]
                except IndexError:
                    node = parent
                else:
                    break
        else:
            node = children[0]