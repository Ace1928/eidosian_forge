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
class RearrangementMorphAction(AATAction):
    staticSize = 4
    actionHeaderSize = 0
    _FLAGS = ['MarkFirst', 'DontAdvance', 'MarkLast']
    _VERBS = {0: 'no change', 1: 'Ax ⇒ xA', 2: 'xD ⇒ Dx', 3: 'AxD ⇒ DxA', 4: 'ABx ⇒ xAB', 5: 'ABx ⇒ xBA', 6: 'xCD ⇒ CDx', 7: 'xCD ⇒ DCx', 8: 'AxCD ⇒ CDxA', 9: 'AxCD ⇒ DCxA', 10: 'ABxD ⇒ DxAB', 11: 'ABxD ⇒ DxBA', 12: 'ABxCD ⇒ CDxAB', 13: 'ABxCD ⇒ CDxBA', 14: 'ABxCD ⇒ DCxAB', 15: 'ABxCD ⇒ DCxBA'}

    def __init__(self):
        self.NewState = 0
        self.Verb = 0
        self.MarkFirst = False
        self.DontAdvance = False
        self.MarkLast = False
        self.ReservedFlags = 0

    def compile(self, writer, font, actionIndex):
        assert actionIndex is None
        writer.writeUShort(self.NewState)
        assert self.Verb >= 0 and self.Verb <= 15, self.Verb
        flags = self.Verb | self.ReservedFlags
        if self.MarkFirst:
            flags |= 32768
        if self.DontAdvance:
            flags |= 16384
        if self.MarkLast:
            flags |= 8192
        writer.writeUShort(flags)

    def decompile(self, reader, font, actionReader):
        assert actionReader is None
        self.NewState = reader.readUShort()
        flags = reader.readUShort()
        self.Verb = flags & 15
        self.MarkFirst = bool(flags & 32768)
        self.DontAdvance = bool(flags & 16384)
        self.MarkLast = bool(flags & 8192)
        self.ReservedFlags = flags & 8176

    def toXML(self, xmlWriter, font, attrs, name):
        xmlWriter.begintag(name, **attrs)
        xmlWriter.newline()
        xmlWriter.simpletag('NewState', value=self.NewState)
        xmlWriter.newline()
        self._writeFlagsToXML(xmlWriter)
        xmlWriter.simpletag('Verb', value=self.Verb)
        verbComment = self._VERBS.get(self.Verb)
        if verbComment is not None:
            xmlWriter.comment(verbComment)
        xmlWriter.newline()
        xmlWriter.endtag(name)
        xmlWriter.newline()

    def fromXML(self, name, attrs, content, font):
        self.NewState = self.Verb = self.ReservedFlags = 0
        self.MarkFirst = self.DontAdvance = self.MarkLast = False
        content = [t for t in content if isinstance(t, tuple)]
        for eltName, eltAttrs, eltContent in content:
            if eltName == 'NewState':
                self.NewState = safeEval(eltAttrs['value'])
            elif eltName == 'Verb':
                self.Verb = safeEval(eltAttrs['value'])
            elif eltName == 'ReservedFlags':
                self.ReservedFlags = safeEval(eltAttrs['value'])
            elif eltName == 'Flags':
                for flag in eltAttrs['value'].split(','):
                    self._setFlag(flag.strip())