import copy
from enum import IntEnum
from functools import reduce
from math import radians
import itertools
from collections import defaultdict, namedtuple
from fontTools.ttLib.tables.otTraverse import dfs_base_table
from fontTools.misc.arrayTools import quantizeRect
from fontTools.misc.roundTools import otRound
from fontTools.misc.transform import Transform, Identity
from fontTools.misc.textTools import bytesjoin, pad, safeEval
from fontTools.pens.boundsPen import ControlBoundsPen
from fontTools.pens.transformPen import TransformPen
from .otBase import (
from fontTools.feaLib.lookupDebugInfo import LookupDebugInfo, LOOKUP_DEBUG_INFO_KEY
import logging
import struct
from typing import TYPE_CHECKING, Iterator, List, Optional, Set
class InsertionMorphAction(AATAction):
    staticSize = 8
    actionHeaderSize = 4
    _FLAGS = ['SetMark', 'DontAdvance', 'CurrentIsKashidaLike', 'MarkedIsKashidaLike', 'CurrentInsertBefore', 'MarkedInsertBefore']

    def __init__(self):
        self.NewState = 0
        for flag in self._FLAGS:
            setattr(self, flag, False)
        self.ReservedFlags = 0
        self.CurrentInsertionAction, self.MarkedInsertionAction = ([], [])

    def compile(self, writer, font, actionIndex):
        assert actionIndex is not None
        writer.writeUShort(self.NewState)
        flags = self.ReservedFlags
        if self.SetMark:
            flags |= 32768
        if self.DontAdvance:
            flags |= 16384
        if self.CurrentIsKashidaLike:
            flags |= 8192
        if self.MarkedIsKashidaLike:
            flags |= 4096
        if self.CurrentInsertBefore:
            flags |= 2048
        if self.MarkedInsertBefore:
            flags |= 1024
        flags |= len(self.CurrentInsertionAction) << 5
        flags |= len(self.MarkedInsertionAction)
        writer.writeUShort(flags)
        if len(self.CurrentInsertionAction) > 0:
            currentIndex = actionIndex[tuple(self.CurrentInsertionAction)]
        else:
            currentIndex = 65535
        writer.writeUShort(currentIndex)
        if len(self.MarkedInsertionAction) > 0:
            markedIndex = actionIndex[tuple(self.MarkedInsertionAction)]
        else:
            markedIndex = 65535
        writer.writeUShort(markedIndex)

    def decompile(self, reader, font, actionReader):
        assert actionReader is not None
        self.NewState = reader.readUShort()
        flags = reader.readUShort()
        self.SetMark = bool(flags & 32768)
        self.DontAdvance = bool(flags & 16384)
        self.CurrentIsKashidaLike = bool(flags & 8192)
        self.MarkedIsKashidaLike = bool(flags & 4096)
        self.CurrentInsertBefore = bool(flags & 2048)
        self.MarkedInsertBefore = bool(flags & 1024)
        self.CurrentInsertionAction = self._decompileInsertionAction(actionReader, font, index=reader.readUShort(), count=(flags & 992) >> 5)
        self.MarkedInsertionAction = self._decompileInsertionAction(actionReader, font, index=reader.readUShort(), count=flags & 31)

    def _decompileInsertionAction(self, actionReader, font, index, count):
        if index == 65535 or count == 0:
            return []
        reader = actionReader.getSubReader(actionReader.pos + index * 2)
        return font.getGlyphNameMany(reader.readUShortArray(count))

    def toXML(self, xmlWriter, font, attrs, name):
        xmlWriter.begintag(name, **attrs)
        xmlWriter.newline()
        xmlWriter.simpletag('NewState', value=self.NewState)
        xmlWriter.newline()
        self._writeFlagsToXML(xmlWriter)
        for g in self.CurrentInsertionAction:
            xmlWriter.simpletag('CurrentInsertionAction', glyph=g)
            xmlWriter.newline()
        for g in self.MarkedInsertionAction:
            xmlWriter.simpletag('MarkedInsertionAction', glyph=g)
            xmlWriter.newline()
        xmlWriter.endtag(name)
        xmlWriter.newline()

    def fromXML(self, name, attrs, content, font):
        self.__init__()
        content = [t for t in content if isinstance(t, tuple)]
        for eltName, eltAttrs, eltContent in content:
            if eltName == 'NewState':
                self.NewState = safeEval(eltAttrs['value'])
            elif eltName == 'Flags':
                for flag in eltAttrs['value'].split(','):
                    self._setFlag(flag.strip())
            elif eltName == 'CurrentInsertionAction':
                self.CurrentInsertionAction.append(eltAttrs['glyph'])
            elif eltName == 'MarkedInsertionAction':
                self.MarkedInsertionAction.append(eltAttrs['glyph'])
            else:
                assert False, eltName

    @staticmethod
    def compileActions(font, states):
        actions, actionIndex, result = (set(), {}, b'')
        for state in states:
            for _glyphClass, trans in state.Transitions.items():
                if trans.CurrentInsertionAction is not None:
                    actions.add(tuple(trans.CurrentInsertionAction))
                if trans.MarkedInsertionAction is not None:
                    actions.add(tuple(trans.MarkedInsertionAction))
        for action in sorted(actions, key=lambda x: (-len(x), x)):
            if action in actionIndex:
                continue
            for start in range(0, len(action)):
                startIndex = len(result) // 2 + start
                for limit in range(start, len(action)):
                    glyphs = action[start:limit + 1]
                    actionIndex.setdefault(glyphs, startIndex)
            for glyph in action:
                glyphID = font.getGlyphID(glyph)
                result += struct.pack('>H', glyphID)
        return (result, actionIndex)