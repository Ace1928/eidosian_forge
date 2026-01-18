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
class _BPlusTreeFormatter:
    signature = 2026540177

    def __init__(self, byteorder='='):
        self.formatter_header = struct.Struct(byteorder + 'IIIIQxxxxxxxx')
        self.formatter_node = struct.Struct(byteorder + '?xH')
        self.fmt_nonleaf = byteorder + '{keySize}sQ'
        self.byteorder = byteorder

    def read(self, stream):
        byteorder = self.byteorder
        formatter = self.formatter_header
        data = stream.read(formatter.size)
        magic, blockSize, keySize, valSize, itemCount = formatter.unpack(data)
        assert magic == _BPlusTreeFormatter.signature
        formatter_node = self.formatter_node
        formatter_nonleaf = struct.Struct(self.fmt_nonleaf.format(keySize=keySize))
        formatter_leaf = struct.Struct(f'{byteorder}{keySize}sII')
        assert keySize == formatter_leaf.size - valSize
        assert valSize == 8
        Node = namedtuple('Node', ['parent', 'children'])
        targets = []
        node = None
        while True:
            data = stream.read(formatter_node.size)
            isLeaf, count = formatter_node.unpack(data)
            if isLeaf:
                for i in range(count):
                    data = stream.read(formatter_leaf.size)
                    key, chromId, chromSize = formatter_leaf.unpack(data)
                    name = key.rstrip(b'\x00').decode()
                    assert chromId == len(targets)
                    sequence = Seq(None, length=chromSize)
                    record = SeqRecord(sequence, id=name)
                    targets.append(record)
            else:
                children = []
                for i in range(count):
                    data = stream.read(formatter_nonleaf.size)
                    key, pos = formatter_nonleaf.unpack(data)
                    children.append(pos)
                parent = node
                node = Node(parent, children)
            while True:
                if node is None:
                    assert len(targets) == itemCount
                    return targets
                children = node.children
                try:
                    pos = children.pop(0)
                except IndexError:
                    node = node.parent
                else:
                    break
            stream.seek(pos)

    def write(self, items, blockSize, output):
        signature = _BPlusTreeFormatter.signature
        keySize = items.dtype['name'].itemsize
        valSize = items.itemsize - keySize
        itemCount = len(items)
        formatter = self.formatter_header
        data = formatter.pack(signature, blockSize, keySize, valSize, itemCount)
        output.write(data)
        formatter_node = self.formatter_node
        formatter_nonleaf = struct.Struct(self.fmt_nonleaf.format(keySize=keySize))
        levels = 1
        while itemCount > blockSize:
            itemCount = len(range(0, itemCount, blockSize))
            levels += 1
        itemCount = len(items)
        bytesInIndexBlock = formatter_node.size + blockSize * formatter_nonleaf.size
        bytesInLeafBlock = formatter_node.size + blockSize * items.itemsize
        isLeaf = False
        indexOffset = output.tell()
        for level in range(levels - 1, 0, -1):
            slotSizePer = blockSize ** level
            nodeSizePer = slotSizePer * blockSize
            indices = range(0, itemCount, nodeSizePer)
            if level == 1:
                bytesInNextLevelBlock = bytesInLeafBlock
            else:
                bytesInNextLevelBlock = bytesInIndexBlock
            levelSize = len(indices) * bytesInIndexBlock
            endLevel = indexOffset + levelSize
            nextChild = endLevel
            for index in indices:
                block = items[index:index + nodeSizePer:slotSizePer]
                n = len(block)
                output.write(formatter_node.pack(isLeaf, n))
                for item in block:
                    data = formatter_nonleaf.pack(item['name'], nextChild)
                    output.write(data)
                    nextChild += bytesInNextLevelBlock
                data = bytes((blockSize - n) * formatter_nonleaf.size)
                output.write(data)
            indexOffset = endLevel
        isLeaf = True
        for index in itertools.count(0, blockSize):
            block = items[index:index + blockSize]
            n = len(block)
            if n == 0:
                break
            output.write(formatter_node.pack(isLeaf, n))
            block.tofile(output)
            data = bytes((blockSize - n) * items.itemsize)
            output.write(data)