from fontTools.config import OPTIONS
from fontTools.misc.textTools import Tag, bytesjoin
from .DefaultTable import DefaultTable
from enum import IntEnum
import sys
import array
import struct
import logging
from functools import lru_cache
from typing import Iterator, NamedTuple, Optional, Tuple
class OTTableWriter(object):
    """Helper class to gather and assemble data for OpenType tables."""

    def __init__(self, localState=None, tableTag=None):
        self.items = []
        self.pos = None
        self.localState = localState
        self.tableTag = tableTag
        self.parent = None

    def __setitem__(self, name, value):
        state = self.localState.copy() if self.localState else dict()
        state[name] = value
        self.localState = state

    def __getitem__(self, name):
        return self.localState[name]

    def __delitem__(self, name):
        del self.localState[name]

    def getDataLength(self):
        """Return the length of this table in bytes, without subtables."""
        l = 0
        for item in self.items:
            if hasattr(item, 'getCountData'):
                l += item.size
            elif hasattr(item, 'subWriter'):
                l += item.offsetSize
            else:
                l = l + len(item)
        return l

    def getData(self):
        """Assemble the data for this writer/table, without subtables."""
        items = list(self.items)
        pos = self.pos
        numItems = len(items)
        for i in range(numItems):
            item = items[i]
            if hasattr(item, 'subWriter'):
                if item.offsetSize == 4:
                    items[i] = packULong(item.subWriter.pos - pos)
                elif item.offsetSize == 2:
                    try:
                        items[i] = packUShort(item.subWriter.pos - pos)
                    except struct.error:
                        overflowErrorRecord = self.getOverflowErrorRecord(item.subWriter)
                        raise OTLOffsetOverflowError(overflowErrorRecord)
                elif item.offsetSize == 3:
                    items[i] = packUInt24(item.subWriter.pos - pos)
                else:
                    raise ValueError(item.offsetSize)
        return bytesjoin(items)

    def getDataForHarfbuzz(self):
        """Assemble the data for this writer/table with all offset field set to 0"""
        items = list(self.items)
        packFuncs = {2: packUShort, 3: packUInt24, 4: packULong}
        for i, item in enumerate(items):
            if hasattr(item, 'subWriter'):
                if item.offsetSize in packFuncs:
                    items[i] = packFuncs[item.offsetSize](0)
                else:
                    raise ValueError(item.offsetSize)
        return bytesjoin(items)

    def __hash__(self):
        return hash(self.items)

    def __ne__(self, other):
        result = self.__eq__(other)
        return result if result is NotImplemented else not result

    def __eq__(self, other):
        if type(self) != type(other):
            return NotImplemented
        return self.items == other.items

    def _doneWriting(self, internedTables, shareExtension=False):
        isExtension = hasattr(self, 'Extension')
        dontShare = hasattr(self, 'DontShare')
        if isExtension and (not shareExtension):
            internedTables = {}
        items = self.items
        for i in range(len(items)):
            item = items[i]
            if hasattr(item, 'getCountData'):
                items[i] = item.getCountData()
            elif hasattr(item, 'subWriter'):
                item.subWriter._doneWriting(internedTables, shareExtension=shareExtension)
                if not dontShare:
                    items[i].subWriter = internedTables.setdefault(item.subWriter, item.subWriter)
        self.items = tuple(items)

    def _gatherTables(self, tables, extTables, done):
        done[id(self)] = True
        numItems = len(self.items)
        iRange = list(range(numItems))
        iRange.reverse()
        isExtension = hasattr(self, 'Extension')
        selfTables = tables
        if isExtension:
            assert extTables is not None, 'Program or XML editing error. Extension subtables cannot contain extensions subtables'
            tables, extTables, done = (extTables, None, {})
        sortCoverageLast = False
        if hasattr(self, 'sortCoverageLast'):
            for i in range(numItems):
                item = self.items[i]
                if hasattr(item, 'subWriter') and getattr(item.subWriter, 'name', None) == 'Coverage':
                    sortCoverageLast = True
                    break
            if id(item.subWriter) not in done:
                item.subWriter._gatherTables(tables, extTables, done)
            else:
                pass
        for i in iRange:
            item = self.items[i]
            if not hasattr(item, 'subWriter'):
                continue
            if sortCoverageLast and i == 1 and (getattr(item.subWriter, 'name', None) == 'Coverage'):
                continue
            if id(item.subWriter) not in done:
                item.subWriter._gatherTables(tables, extTables, done)
            else:
                pass
        selfTables.append(self)

    def _gatherGraphForHarfbuzz(self, tables, obj_list, done, objidx, virtual_edges):
        real_links = []
        virtual_links = []
        item_idx = objidx
        for idx in virtual_edges:
            virtual_links.append((0, 0, idx))
        sortCoverageLast = False
        coverage_idx = 0
        if hasattr(self, 'sortCoverageLast'):
            for i, item in enumerate(self.items):
                if getattr(item, 'name', None) == 'Coverage':
                    sortCoverageLast = True
                    if id(item) not in done:
                        coverage_idx = item_idx = item._gatherGraphForHarfbuzz(tables, obj_list, done, item_idx, virtual_edges)
                    else:
                        coverage_idx = done[id(item)]
                    virtual_edges.append(coverage_idx)
                    break
        child_idx = 0
        offset_pos = 0
        for i, item in enumerate(self.items):
            if hasattr(item, 'subWriter'):
                pos = offset_pos
            elif hasattr(item, 'getCountData'):
                offset_pos += item.size
                continue
            else:
                offset_pos = offset_pos + len(item)
                continue
            if id(item.subWriter) not in done:
                child_idx = item_idx = item.subWriter._gatherGraphForHarfbuzz(tables, obj_list, done, item_idx, virtual_edges)
            else:
                child_idx = done[id(item.subWriter)]
            real_edge = (pos, item.offsetSize, child_idx)
            real_links.append(real_edge)
            offset_pos += item.offsetSize
        tables.append(self)
        obj_list.append((real_links, virtual_links))
        item_idx += 1
        done[id(self)] = item_idx
        if sortCoverageLast:
            virtual_edges.pop()
        return item_idx

    def getAllDataUsingHarfbuzz(self, tableTag):
        """The Whole table is represented as a Graph.
        Assemble graph data and call Harfbuzz repacker to pack the table.
        Harfbuzz repacker is faster and retain as much sub-table sharing as possible, see also:
        https://github.com/harfbuzz/harfbuzz/blob/main/docs/repacker.md
        The input format for hb.repack() method is explained here:
        https://github.com/harfbuzz/uharfbuzz/blob/main/src/uharfbuzz/_harfbuzz.pyx#L1149
        """
        internedTables = {}
        self._doneWriting(internedTables, shareExtension=True)
        tables = []
        obj_list = []
        done = {}
        objidx = 0
        virtual_edges = []
        self._gatherGraphForHarfbuzz(tables, obj_list, done, objidx, virtual_edges)
        pos = 0
        for table in tables:
            table.pos = pos
            pos = pos + table.getDataLength()
        data = []
        for table in tables:
            tableData = table.getDataForHarfbuzz()
            data.append(tableData)
        if hasattr(hb, 'repack_with_tag'):
            return hb.repack_with_tag(str(tableTag), data, obj_list)
        else:
            return hb.repack(data, obj_list)

    def getAllData(self, remove_duplicate=True):
        """Assemble all data, including all subtables."""
        if remove_duplicate:
            internedTables = {}
            self._doneWriting(internedTables)
        tables = []
        extTables = []
        done = {}
        self._gatherTables(tables, extTables, done)
        tables.reverse()
        extTables.reverse()
        pos = 0
        for table in tables:
            table.pos = pos
            pos = pos + table.getDataLength()
        for table in extTables:
            table.pos = pos
            pos = pos + table.getDataLength()
        data = []
        for table in tables:
            tableData = table.getData()
            data.append(tableData)
        for table in extTables:
            tableData = table.getData()
            data.append(tableData)
        return bytesjoin(data)

    def getSubWriter(self):
        subwriter = self.__class__(self.localState, self.tableTag)
        subwriter.parent = self
        return subwriter

    def writeValue(self, typecode, value):
        self.items.append(struct.pack(f'>{typecode}', value))

    def writeArray(self, typecode, values):
        a = array.array(typecode, values)
        if sys.byteorder != 'big':
            a.byteswap()
        self.items.append(a.tobytes())

    def writeInt8(self, value):
        assert -128 <= value < 128, value
        self.items.append(struct.pack('>b', value))

    def writeInt8Array(self, values):
        self.writeArray('b', values)

    def writeShort(self, value):
        assert -32768 <= value < 32768, value
        self.items.append(struct.pack('>h', value))

    def writeShortArray(self, values):
        self.writeArray('h', values)

    def writeLong(self, value):
        self.items.append(struct.pack('>i', value))

    def writeLongArray(self, values):
        self.writeArray('i', values)

    def writeUInt8(self, value):
        assert 0 <= value < 256, value
        self.items.append(struct.pack('>B', value))

    def writeUInt8Array(self, values):
        self.writeArray('B', values)

    def writeUShort(self, value):
        assert 0 <= value < 65536, value
        self.items.append(struct.pack('>H', value))

    def writeUShortArray(self, values):
        self.writeArray('H', values)

    def writeULong(self, value):
        self.items.append(struct.pack('>I', value))

    def writeULongArray(self, values):
        self.writeArray('I', values)

    def writeUInt24(self, value):
        assert 0 <= value < 16777216, value
        b = struct.pack('>L', value)
        self.items.append(b[1:])

    def writeUInt24Array(self, values):
        for value in values:
            self.writeUInt24(value)

    def writeTag(self, tag):
        tag = Tag(tag).tobytes()
        assert len(tag) == 4, tag
        self.items.append(tag)

    def writeSubTable(self, subWriter, offsetSize):
        self.items.append(OffsetToWriter(subWriter, offsetSize))

    def writeCountReference(self, table, name, size=2, value=None):
        ref = CountReference(table, name, size=size, value=value)
        self.items.append(ref)
        return ref

    def writeStruct(self, format, values):
        data = struct.pack(*(format,) + values)
        self.items.append(data)

    def writeData(self, data):
        self.items.append(data)

    def getOverflowErrorRecord(self, item):
        LookupListIndex = SubTableIndex = itemName = itemIndex = None
        if self.name == 'LookupList':
            LookupListIndex = item.repeatIndex
        elif self.name == 'Lookup':
            LookupListIndex = self.repeatIndex
            SubTableIndex = item.repeatIndex
        else:
            itemName = getattr(item, 'name', '<none>')
            if hasattr(item, 'repeatIndex'):
                itemIndex = item.repeatIndex
            if self.name == 'SubTable':
                LookupListIndex = self.parent.repeatIndex
                SubTableIndex = self.repeatIndex
            elif self.name == 'ExtSubTable':
                LookupListIndex = self.parent.parent.repeatIndex
                SubTableIndex = self.parent.repeatIndex
            else:
                itemName = '.'.join([self.name, itemName])
                p1 = self.parent
                while p1 and p1.name not in ['ExtSubTable', 'SubTable']:
                    itemName = '.'.join([p1.name, itemName])
                    p1 = p1.parent
                if p1:
                    if p1.name == 'ExtSubTable':
                        LookupListIndex = p1.parent.parent.repeatIndex
                        SubTableIndex = p1.parent.repeatIndex
                    else:
                        LookupListIndex = p1.parent.repeatIndex
                        SubTableIndex = p1.repeatIndex
        return OverflowErrorRecord((self.tableTag, LookupListIndex, SubTableIndex, itemName, itemIndex))