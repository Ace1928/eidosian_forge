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
class LookupList(BaseTable):

    @property
    def table(self):
        for l in self.Lookup:
            for st in l.SubTable:
                if type(st).__name__.endswith('Subst'):
                    return 'GSUB'
                if type(st).__name__.endswith('Pos'):
                    return 'GPOS'
        raise ValueError

    def toXML2(self, xmlWriter, font):
        if not font or 'Debg' not in font or LOOKUP_DEBUG_INFO_KEY not in font['Debg'].data:
            return super().toXML2(xmlWriter, font)
        debugData = font['Debg'].data[LOOKUP_DEBUG_INFO_KEY][self.table]
        for conv in self.getConverters():
            if conv.repeat:
                value = getattr(self, conv.name, [])
                for lookupIndex, item in enumerate(value):
                    if str(lookupIndex) in debugData:
                        info = LookupDebugInfo(*debugData[str(lookupIndex)])
                        tag = info.location
                        if info.name:
                            tag = f'{info.name}: {tag}'
                        if info.feature:
                            script, language, feature = info.feature
                            tag = f'{tag} in {feature} ({script}/{language})'
                        xmlWriter.comment(tag)
                        xmlWriter.newline()
                    conv.xmlWrite(xmlWriter, font, item, conv.name, [('index', lookupIndex)])
            else:
                if conv.aux and (not eval(conv.aux, None, vars(self))):
                    continue
                value = getattr(self, conv.name, None)
                conv.xmlWrite(xmlWriter, font, value, conv.name, [])