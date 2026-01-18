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
def _search_index(self, stream, chromIx, start, end):
    formatter = struct.Struct(self.byteorder + 'III')
    size = formatter.size
    padded_start = start - 1
    padded_end = end + 1
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
                child_chromIx, child_chromStart, child_chromEnd = formatter.unpack(data[:size])
                rest, data = data[size:].split(b'\x00', 1)
                if child_chromIx != chromIx:
                    continue
                if end <= child_chromStart or child_chromEnd <= start:
                    if child_chromStart != child_chromEnd:
                        continue
                    if child_chromStart != end and child_chromEnd != start:
                        continue
                yield (child_chromIx, child_chromStart, child_chromEnd, rest)
        else:
            visit_child = False
            for child in children:
                if (child.endChromIx, child.endBase) < (chromIx, padded_start):
                    continue
                if (chromIx, padded_end) < (child.startChromIx, child.startBase):
                    continue
                visit_child = True
                break
            if visit_child:
                node = child
                continue
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