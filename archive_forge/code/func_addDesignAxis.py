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
def addDesignAxis(self, designAxis, location):
    if 'DesignAxes' not in self.stat_:
        self.stat_['DesignAxes'] = []
    if designAxis.tag in (r.tag for r in self.stat_['DesignAxes']):
        raise FeatureLibError(f'DesignAxis already defined for tag "{designAxis.tag}".', location)
    if designAxis.axisOrder in (r.axisOrder for r in self.stat_['DesignAxes']):
        raise FeatureLibError(f'DesignAxis already defined for axis number {designAxis.axisOrder}.', location)
    self.stat_['DesignAxes'].append(designAxis)