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
class COLR(BaseTable):

    def decompile(self, reader, font):
        subReader = reader.getSubReader(offset=0)
        for conv in self.getConverters():
            if conv.name != 'LayerRecordCount':
                subReader.advance(conv.staticSize)
                continue
            reader[conv.name] = conv.read(subReader, font, tableDict={})
            break
        else:
            raise AssertionError('LayerRecordCount converter not found')
        return BaseTable.decompile(self, reader, font)

    def preWrite(self, font):
        self.LayerRecordCount = None
        return {**self.__dict__, 'LayerRecordCount': CountReference(self.__dict__, 'LayerRecordCount')}

    def computeClipBoxes(self, glyphSet: '_TTGlyphSet', quantization: int=1):
        if self.Version == 0:
            return
        clips = {}
        for rec in self.BaseGlyphList.BaseGlyphPaintRecord:
            try:
                clipBox = rec.Paint.computeClipBox(self, glyphSet, quantization)
            except Exception as e:
                from fontTools.ttLib import TTLibError
                raise TTLibError(f'Failed to compute COLR ClipBox for {rec.BaseGlyph!r}') from e
            if clipBox is not None:
                clips[rec.BaseGlyph] = clipBox
        hasClipList = hasattr(self, 'ClipList') and self.ClipList is not None
        if not clips:
            if hasClipList:
                self.ClipList = None
        else:
            if not hasClipList:
                self.ClipList = ClipList()
                self.ClipList.Format = 1
            self.ClipList.clips = clips