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
class _TupleVarStoreAdapter(object):

    def __init__(self, regions, axisOrder, tupleVarData, itemCounts):
        self.regions = regions
        self.axisOrder = axisOrder
        self.tupleVarData = tupleVarData
        self.itemCounts = itemCounts

    @classmethod
    def fromItemVarStore(cls, itemVarStore, fvarAxes):
        axisOrder = [axis.axisTag for axis in fvarAxes]
        regions = [region.get_support(fvarAxes) for region in itemVarStore.VarRegionList.Region]
        tupleVarData = []
        itemCounts = []
        for varData in itemVarStore.VarData:
            variations = []
            varDataRegions = (regions[i] for i in varData.VarRegionIndex)
            for axes, coordinates in zip(varDataRegions, zip(*varData.Item)):
                variations.append(TupleVariation(axes, list(coordinates)))
            tupleVarData.append(variations)
            itemCounts.append(varData.ItemCount)
        return cls(regions, axisOrder, tupleVarData, itemCounts)

    def rebuildRegions(self):
        uniqueRegions = collections.OrderedDict.fromkeys((frozenset(var.axes.items()) for variations in self.tupleVarData for var in variations))
        newRegions = []
        for region in self.regions:
            regionAxes = frozenset(region.items())
            if regionAxes in uniqueRegions:
                newRegions.append(region)
                del uniqueRegions[regionAxes]
        if uniqueRegions:
            newRegions.extend((dict(region) for region in uniqueRegions))
        self.regions = newRegions

    def instantiate(self, axisLimits):
        defaultDeltaArray = []
        for variations, itemCount in zip(self.tupleVarData, self.itemCounts):
            defaultDeltas = instantiateTupleVariationStore(variations, axisLimits)
            if not defaultDeltas:
                defaultDeltas = [0] * itemCount
            defaultDeltaArray.append(defaultDeltas)
        self.rebuildRegions()
        pinnedAxes = set(axisLimits.pinnedLocation())
        self.axisOrder = [axisTag for axisTag in self.axisOrder if axisTag not in pinnedAxes]
        return defaultDeltaArray

    def asItemVarStore(self):
        regionOrder = [frozenset(axes.items()) for axes in self.regions]
        varDatas = []
        for variations, itemCount in zip(self.tupleVarData, self.itemCounts):
            if variations:
                assert len(variations[0].coordinates) == itemCount
                varRegionIndices = [regionOrder.index(frozenset(var.axes.items())) for var in variations]
                varDataItems = list(zip(*(var.coordinates for var in variations)))
                varDatas.append(builder.buildVarData(varRegionIndices, varDataItems, optimize=False))
            else:
                varDatas.append(builder.buildVarData([], [[] for _ in range(itemCount)]))
        regionList = builder.buildVarRegionList(self.regions, self.axisOrder)
        itemVarStore = builder.buildVarStore(regionList, varDatas)
        itemVarStore.prune_regions()
        return itemVarStore