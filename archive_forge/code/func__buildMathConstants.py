from collections import namedtuple, OrderedDict
import os
from fontTools.misc.fixedTools import fixedToFloat
from fontTools.misc.roundTools import otRound
from fontTools import ttLib
from fontTools.ttLib.tables import otTables as ot
from fontTools.ttLib.tables.otBase import (
from fontTools.ttLib.tables import otBase
from fontTools.feaLib.ast import STATNameStatement
from fontTools.otlLib.optimize.gpos import (
from fontTools.otlLib.error import OpenTypeLibError
from functools import reduce
import logging
import copy
def _buildMathConstants(constants):
    if not constants:
        return None
    mathConstants = ot.MathConstants()
    for conv in mathConstants.getConverters():
        value = otRound(constants.get(conv.name, 0))
        if conv.tableClass:
            assert issubclass(conv.tableClass, ot.MathValueRecord)
            value = _mathValueRecord(value)
        setattr(mathConstants, conv.name, value)
    return mathConstants