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
def _buildClasses():
    import re
    from .otData import otData
    formatPat = re.compile('([A-Za-z0-9]+)Format(\\d+)$')
    namespace = globals()
    for name, table in otData:
        baseClass = BaseTable
        m = formatPat.match(name)
        if m:
            name = m.group(1)
            formatType = table[0][0]
            baseClass = getFormatSwitchingBaseTableClass(formatType)
        if name not in namespace:
            cls = type(name, (baseClass,), {})
            if name in ('GSUB', 'GPOS'):
                cls.DontShare = True
            namespace[name] = cls
    for name, _ in otData:
        if name.startswith('Var') and len(name) > 3 and (name[3:] in namespace):
            varType = namespace[name]
            noVarType = namespace[name[3:]]
            varType.NoVarType = noVarType
            noVarType.VarType = varType
    for base, alts in _equivalents.items():
        base = namespace[base]
        for alt in alts:
            namespace[alt] = base
    global lookupTypes
    lookupTypes = {'GSUB': {1: SingleSubst, 2: MultipleSubst, 3: AlternateSubst, 4: LigatureSubst, 5: ContextSubst, 6: ChainContextSubst, 7: ExtensionSubst, 8: ReverseChainSingleSubst}, 'GPOS': {1: SinglePos, 2: PairPos, 3: CursivePos, 4: MarkBasePos, 5: MarkLigPos, 6: MarkMarkPos, 7: ContextPos, 8: ChainContextPos, 9: ExtensionPos}, 'mort': {4: NoncontextualMorph}, 'morx': {0: RearrangementMorph, 1: ContextualMorph, 2: LigatureMorph, 4: NoncontextualMorph, 5: InsertionMorph}}
    lookupTypes['JSTF'] = lookupTypes['GPOS']
    for lookupEnum in lookupTypes.values():
        for enum, cls in lookupEnum.items():
            cls.LookupType = enum
    global featureParamTypes
    featureParamTypes = {'size': FeatureParamsSize}
    for i in range(1, 20 + 1):
        featureParamTypes['ss%02d' % i] = FeatureParamsStylisticSet
    for i in range(1, 99 + 1):
        featureParamTypes['cv%02d' % i] = FeatureParamsCharacterVariants
    from .otConverters import buildConverters
    for name, table in otData:
        m = formatPat.match(name)
        if m:
            name, format = m.groups()
            format = int(format)
            cls = namespace[name]
            if not hasattr(cls, 'converters'):
                cls.converters = {}
                cls.convertersByName = {}
            converters, convertersByName = buildConverters(table[1:], namespace)
            cls.converters[format] = converters
            cls.convertersByName[format] = convertersByName
        else:
            cls = namespace[name]
            cls.converters, cls.convertersByName = buildConverters(table, namespace)