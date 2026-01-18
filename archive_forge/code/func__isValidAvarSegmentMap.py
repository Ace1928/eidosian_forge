from fontTools.misc.fixedTools import (
from fontTools.varLib.models import normalizeValue, piecewiseLinearMap
from fontTools.ttLib import TTFont
from fontTools.ttLib.tables.TupleVariation import TupleVariation
from fontTools.ttLib.tables import _g_l_y_f
from fontTools import varLib
from fontTools import subset  # noqa: F401
from fontTools.varLib import builder
from fontTools.varLib.mvar import MVAR_ENTRIES
from fontTools.varLib.merger import MutatorMerger
from fontTools.varLib.instancer import names
from .featureVars import instantiateFeatureVariations
from fontTools.misc.cliTools import makeOutputFileName
from fontTools.varLib.instancer import solver
import collections
import dataclasses
from contextlib import contextmanager
from copy import deepcopy
from enum import IntEnum
import logging
import os
import re
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple, Union
import warnings
def _isValidAvarSegmentMap(axisTag, segmentMap):
    if not segmentMap:
        return True
    if not {(-1.0, -1.0), (0, 0), (1.0, 1.0)}.issubset(segmentMap.items()):
        log.warning(f"Invalid avar SegmentMap record for axis '{axisTag}': does not include all required value maps {{-1.0: -1.0, 0: 0, 1.0: 1.0}}")
        return False
    previousValue = None
    for fromCoord, toCoord in sorted(segmentMap.items()):
        if previousValue is not None and previousValue > toCoord:
            log.warning(f"Invalid avar AxisValueMap({fromCoord}, {toCoord}) record for axis '{axisTag}': the toCoordinate value must be >= to the toCoordinate value of the preceding record ({previousValue}).")
            return False
        previousValue = toCoord
    return True