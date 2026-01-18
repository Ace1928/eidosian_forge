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
class _RedBlackTreeNode:
    __slots__ = ('left', 'right', 'color', 'item')

    def traverse(self):
        if self.left is not None:
            yield from self.left.traverse()
        yield self.item
        if self.right is not None:
            yield from self.right.traverse()

    def traverse_range(self, start, end):
        if self.item.end <= start:
            if self.right is not None:
                yield from self.right.traverse_range(start, end)
        elif end <= self.item.start:
            if self.left is not None:
                yield from self.left.traverse_range(start, end)
        else:
            if self.left is not None:
                yield from self.left.traverse_range(start, end)
            yield self.item
            if self.right is not None:
                yield from self.right.traverse_range(start, end)