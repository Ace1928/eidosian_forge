from fontTools.misc.fixedTools import (
from fontTools.misc.roundTools import nearestMultipleShortestRepr, otRound
from fontTools.misc.textTools import bytesjoin, tobytes, tostr, pad, safeEval
from fontTools.ttLib import getSearchRange
from .otBase import (
from .otTables import (
from itertools import zip_longest
from functools import partial
import re
import struct
from typing import Optional
import logging
class MorxSubtableConverter(BaseConverter):
    _PROCESSING_ORDERS = {(False, False): 'LayoutOrder', (True, False): 'ReversedLayoutOrder', (False, True): 'LogicalOrder', (True, True): 'ReversedLogicalOrder'}
    _PROCESSING_ORDERS_REVERSED = {val: key for key, val in _PROCESSING_ORDERS.items()}

    def __init__(self, name, repeat, aux, tableClass=None, *, description=''):
        BaseConverter.__init__(self, name, repeat, aux, tableClass, description=description)

    def _setTextDirectionFromCoverageFlags(self, flags, subtable):
        if flags & 32 != 0:
            subtable.TextDirection = 'Any'
        elif flags & 128 != 0:
            subtable.TextDirection = 'Vertical'
        else:
            subtable.TextDirection = 'Horizontal'

    def read(self, reader, font, tableDict):
        pos = reader.pos
        m = MorxSubtable()
        m.StructLength = reader.readULong()
        flags = reader.readUInt8()
        orderKey = (flags & 64 != 0, flags & 16 != 0)
        m.ProcessingOrder = self._PROCESSING_ORDERS[orderKey]
        self._setTextDirectionFromCoverageFlags(flags, m)
        m.Reserved = reader.readUShort()
        m.Reserved |= (flags & 15) << 16
        m.MorphType = reader.readUInt8()
        m.SubFeatureFlags = reader.readULong()
        tableClass = lookupTypes['morx'].get(m.MorphType)
        if tableClass is None:
            assert False, "unsupported 'morx' lookup type %s" % m.MorphType
        headerLength = reader.pos - pos
        data = reader.data[reader.pos:reader.pos + m.StructLength - headerLength]
        assert len(data) == m.StructLength - headerLength
        subReader = OTTableReader(data=data, tableTag=reader.tableTag)
        m.SubStruct = tableClass()
        m.SubStruct.decompile(subReader, font)
        reader.seek(pos + m.StructLength)
        return m

    def xmlWrite(self, xmlWriter, font, value, name, attrs):
        xmlWriter.begintag(name, attrs)
        xmlWriter.newline()
        xmlWriter.comment('StructLength=%d' % value.StructLength)
        xmlWriter.newline()
        xmlWriter.simpletag('TextDirection', value=value.TextDirection)
        xmlWriter.newline()
        xmlWriter.simpletag('ProcessingOrder', value=value.ProcessingOrder)
        xmlWriter.newline()
        if value.Reserved != 0:
            xmlWriter.simpletag('Reserved', value='0x%04x' % value.Reserved)
            xmlWriter.newline()
        xmlWriter.comment('MorphType=%d' % value.MorphType)
        xmlWriter.newline()
        xmlWriter.simpletag('SubFeatureFlags', value='0x%08x' % value.SubFeatureFlags)
        xmlWriter.newline()
        value.SubStruct.toXML(xmlWriter, font)
        xmlWriter.endtag(name)
        xmlWriter.newline()

    def xmlRead(self, attrs, content, font):
        m = MorxSubtable()
        covFlags = 0
        m.Reserved = 0
        for eltName, eltAttrs, eltContent in filter(istuple, content):
            if eltName == 'CoverageFlags':
                covFlags = safeEval(eltAttrs['value'])
                orderKey = (covFlags & 64 != 0, covFlags & 16 != 0)
                m.ProcessingOrder = self._PROCESSING_ORDERS[orderKey]
                self._setTextDirectionFromCoverageFlags(covFlags, m)
            elif eltName == 'ProcessingOrder':
                m.ProcessingOrder = eltAttrs['value']
                assert m.ProcessingOrder in self._PROCESSING_ORDERS_REVERSED, 'unknown ProcessingOrder: %s' % m.ProcessingOrder
            elif eltName == 'TextDirection':
                m.TextDirection = eltAttrs['value']
                assert m.TextDirection in {'Horizontal', 'Vertical', 'Any'}, 'unknown TextDirection %s' % m.TextDirection
            elif eltName == 'Reserved':
                m.Reserved = safeEval(eltAttrs['value'])
            elif eltName == 'SubFeatureFlags':
                m.SubFeatureFlags = safeEval(eltAttrs['value'])
            elif eltName.endswith('Morph'):
                m.fromXML(eltName, eltAttrs, eltContent, font)
            else:
                assert False, eltName
        m.Reserved = (covFlags & 15) << 16 | m.Reserved
        return m

    def write(self, writer, font, tableDict, value, repeatIndex=None):
        covFlags = (value.Reserved & 983040) >> 16
        reverseOrder, logicalOrder = self._PROCESSING_ORDERS_REVERSED[value.ProcessingOrder]
        covFlags |= 128 if value.TextDirection == 'Vertical' else 0
        covFlags |= 64 if reverseOrder else 0
        covFlags |= 32 if value.TextDirection == 'Any' else 0
        covFlags |= 16 if logicalOrder else 0
        value.CoverageFlags = covFlags
        lengthIndex = len(writer.items)
        before = writer.getDataLength()
        value.StructLength = 3735928559
        origReserved = value.Reserved
        value.Reserved = value.Reserved & 65535
        value.compile(writer, font)
        value.Reserved = origReserved
        assert writer.items[lengthIndex] == b'\xde\xad\xbe\xef'
        length = writer.getDataLength() - before
        writer.items[lengthIndex] = struct.pack('>L', length)