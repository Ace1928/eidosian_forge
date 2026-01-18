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
class Struct(BaseConverter):

    def getRecordSize(self, reader):
        return self.tableClass and self.tableClass.getRecordSize(reader)

    def read(self, reader, font, tableDict):
        table = self.tableClass()
        table.decompile(reader, font)
        return table

    def write(self, writer, font, tableDict, value, repeatIndex=None):
        value.compile(writer, font)

    def xmlWrite(self, xmlWriter, font, value, name, attrs):
        if value is None:
            if attrs:
                xmlWriter.simpletag(name, attrs + [('empty', 1)])
                xmlWriter.newline()
            else:
                pass
        else:
            value.toXML(xmlWriter, font, attrs, name=name)

    def xmlRead(self, attrs, content, font):
        if 'empty' in attrs and safeEval(attrs['empty']):
            return None
        table = self.tableClass()
        Format = attrs.get('Format')
        if Format is not None:
            table.Format = int(Format)
        noPostRead = not hasattr(table, 'postRead')
        if noPostRead:
            cleanPropagation = False
            for conv in table.getConverters():
                if conv.isPropagated:
                    cleanPropagation = True
                    if not hasattr(font, '_propagator'):
                        font._propagator = {}
                    propagator = font._propagator
                    assert conv.name not in propagator, (conv.name, propagator)
                    setattr(table, conv.name, None)
                    propagator[conv.name] = CountReference(table.__dict__, conv.name)
        for element in content:
            if isinstance(element, tuple):
                name, attrs, content = element
                table.fromXML(name, attrs, content, font)
            else:
                pass
        table.populateDefaults(propagator=getattr(font, '_propagator', None))
        if noPostRead:
            if cleanPropagation:
                for conv in table.getConverters():
                    if conv.isPropagated:
                        propagator = font._propagator
                        del propagator[conv.name]
                        if not propagator:
                            del font._propagator
        return table

    def __repr__(self):
        return 'Struct of ' + repr(self.tableClass)