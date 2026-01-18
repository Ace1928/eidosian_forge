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
class _RangeTree:
    __slots__ = ('root', 'n', 'freeList', 'stack', 'chromId', 'chromSize')

    def __init__(self, chromId, chromSize):
        self.root = None
        self.n = 0
        self.freeList = []
        self.chromId = chromId
        self.chromSize = chromSize

    @classmethod
    def generate(cls, chromUsageList, alignments):
        alignments.rewind()
        alignment = None
        for chromName, chromId, chromSize in chromUsageList:
            chromName = chromName.decode()
            tree = _RangeTree(chromId, chromSize)
            if alignment is not None:
                tree.addToCoverageDepth(alignment)
            for alignment in alignments:
                if alignment.target.id != chromName:
                    break
                tree.addToCoverageDepth(alignment)
            yield tree

    def generate_summaries(self, scale, totalSum):
        ranges = self.root.traverse()
        start, end, val = next(ranges)
        chromId = self.chromId
        chromSize = self.chromSize
        summary = _RegionSummary(chromId, start, min(start + scale, chromSize), val)
        while True:
            size = max(end - start, 1)
            totalSum.update(size, val)
            if summary.end <= start:
                summary = _RegionSummary(chromId, start, min(start + scale, chromSize), val)
            while end > summary.end:
                overlap = min(end, summary.end) - max(start, summary.start)
                assert overlap > 0
                summary.update(overlap, val)
                size -= overlap
                start = summary.end
                yield summary
                summary = _RegionSummary(chromId, start, min(start + scale, chromSize), val)
            summary.update(size, val)
            try:
                start, end, val = next(ranges)
            except StopIteration:
                break
            if summary.end <= start:
                yield summary
        yield summary

    def find(self, start, end):
        p = self.root
        while p is not None:
            if end <= p.item.start:
                p = p.left
            elif p.item.end <= start:
                p = p.right
            else:
                return p.item

    def restructure(self, x, y, z):
        tos = len(self.stack)
        if y is x.left:
            if z is y.left:
                midNode = y
                y.left = z
                x.left = y.right
                y.right = x
            else:
                midNode = z
                y.right = z.left
                z.left = y
                x.left = z.right
                z.right = x
        elif z is y.left:
            midNode = z
            x.right = z.left
            z.left = x
            y.left = z.right
            z.right = y
        else:
            midNode = y
            x.right = y.left
            y.left = x
            y.right = z
        if tos != 0:
            parent = self.stack[tos - 1]
            if x is parent.left:
                parent.left = midNode
            else:
                parent.right = midNode
        else:
            self.root = midNode
        return midNode

    def add(self, item):
        self.stack = []
        try:
            x = self.freeList.pop()
        except IndexError:
            x = _RedBlackTreeNode()
        else:
            self.freeList = x.right
        x.left = None
        x.right = None
        x.item = item
        x.color = None
        p = self.root
        if p is not None:
            while True:
                self.stack.append(p)
                if item.end <= p.item.start:
                    p = p.left
                    if p is None:
                        p = self.stack.pop()
                        p.left = x
                        break
                elif p.item.end <= item.start:
                    p = p.right
                    if p is None:
                        p = self.stack.pop()
                        p.right = x
                        break
                else:
                    return
            col = True
        else:
            self.root = x
            col = False
        x.color = col
        self.n += 1
        if len(self.stack) > 0:
            while p.color is True:
                m = self.stack.pop()
                if p == m.left:
                    q = m.right
                else:
                    q = m.left
                if q is None or q.color is False:
                    m = self.restructure(m, p, x)
                    m.color = False
                    m.left.color = True
                    m.right.color = True
                    break
                p.color = False
                q.color = False
                if len(self.stack) == 0:
                    break
                m.color = True
                x = m
                p = self.stack.pop()

    def addToCoverageDepth(self, alignment):
        start = alignment.coordinates[0, 0]
        end = alignment.coordinates[0, -1]
        if start > end:
            start, end = (end, start)
        existing = self.find(start, end)
        if existing is None:
            r = _Range(start, end, val=1)
            self.add(r)
        elif existing.start <= start and existing.end >= end:
            if existing.start < start:
                r = _Range(existing.start, start, existing.val)
                existing.start = start
                self.add(r)
            if existing.end > end:
                r = _Range(end, existing.end, existing.val)
                existing.end = end
                self.add(r)
            existing.val += 1
        else:
            items = list(self.root.traverse_range(start, end))
            s = start
            e = end
            for item in items:
                if s < item.start:
                    r = _Range(s, item.start, 1)
                    s = item.start
                    self.add(r)
                elif s > item.start:
                    r = _Range(item.start, s, item.val)
                    item.start = s
                    self.add(r)
                if item.start < end and item.end > end:
                    r = _Range(end, item.end, item.val)
                    item.end = end
                    self.add(r)
                item.val += 1
                s = item.end
            if s < e:
                r = _Range(s, e, 1)
                self.add(r)