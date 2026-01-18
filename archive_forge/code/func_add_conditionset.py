from fontTools.misc import sstruct
from fontTools.misc.textTools import Tag, tostr, binary2num, safeEval
from fontTools.feaLib.error import FeatureLibError
from fontTools.feaLib.lookupDebugInfo import (
from fontTools.feaLib.parser import Parser
from fontTools.feaLib.ast import FeatureFile
from fontTools.feaLib.variableScalar import VariableScalar
from fontTools.otlLib import builder as otl
from fontTools.otlLib.maxContextCalc import maxCtxFont
from fontTools.ttLib import newTable, getTableModule
from fontTools.ttLib.tables import otBase, otTables
from fontTools.otlLib.builder import (
from fontTools.otlLib.error import OpenTypeLibError
from fontTools.varLib.varStore import OnlineVarStoreBuilder
from fontTools.varLib.builder import buildVarDevTable
from fontTools.varLib.featureVars import addFeatureVariationsRaw
from fontTools.varLib.models import normalizeValue, piecewiseLinearMap
from collections import defaultdict
import copy
import itertools
from io import StringIO
import logging
import warnings
import os
def add_conditionset(self, location, key, value):
    if 'fvar' not in self.font:
        raise FeatureLibError("Cannot add feature variations to a font without an 'fvar' table", location)
    axisMap = {axis.axisTag: (axis.minValue, axis.defaultValue, axis.maxValue) for axis in self.axes}
    value = {tag: (normalizeValue(bottom, axisMap[tag]), normalizeValue(top, axisMap[tag])) for tag, (bottom, top) in value.items()}
    if 'avar' in self.font:
        mapping = self.font['avar'].segments
        value = {axis: tuple((piecewiseLinearMap(v, mapping[axis]) if axis in mapping else v for v in condition_range)) for axis, condition_range in value.items()}
    self.conditionsets_[key] = value