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
def _instantiateGvarGlyph(glyphname, glyf, gvar, hMetrics, vMetrics, axisLimits, optimize=True):
    coordinates, ctrl = glyf._getCoordinatesAndControls(glyphname, hMetrics, vMetrics)
    endPts = ctrl.endPts
    tupleVarStore = gvar.variations.get(glyphname)
    if tupleVarStore:
        defaultDeltas = instantiateTupleVariationStore(tupleVarStore, axisLimits, coordinates, endPts)
        if defaultDeltas:
            coordinates += _g_l_y_f.GlyphCoordinates(defaultDeltas)
    glyph = glyf[glyphname]
    if glyph.isVarComposite():
        for component in glyph.components:
            newLocation = {}
            for tag, loc in component.location.items():
                if tag not in axisLimits:
                    newLocation[tag] = loc
                    continue
                if component.flags & _g_l_y_f.VarComponentFlags.AXES_HAVE_VARIATION:
                    raise NotImplementedError('Instancing accross VarComposite axes with variation is not supported.')
                limits = axisLimits[tag]
                loc = limits.renormalizeValue(loc, extrapolate=False)
                newLocation[tag] = loc
            component.location = newLocation
    glyf._setCoordinates(glyphname, coordinates, hMetrics, vMetrics)
    if not tupleVarStore:
        if glyphname in gvar.variations:
            del gvar.variations[glyphname]
        return
    if optimize:
        isComposite = glyf[glyphname].isComposite()
        for var in tupleVarStore:
            var.optimize(coordinates, endPts, isComposite=isComposite)