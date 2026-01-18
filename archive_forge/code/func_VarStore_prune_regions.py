from fontTools.misc.roundTools import noRound, otRound
from fontTools.misc.intTools import bit_count
from fontTools.ttLib.tables import otTables as ot
from fontTools.varLib.models import supportScalar
from fontTools.varLib.builder import (
from functools import partial
from collections import defaultdict
from heapq import heappush, heappop
def VarStore_prune_regions(self):
    """Remove unused VarRegions."""
    usedRegions = set()
    for data in self.VarData:
        usedRegions.update(data.VarRegionIndex)
    regionList = self.VarRegionList
    regions = regionList.Region
    newRegions = []
    regionMap = {}
    for i in sorted(usedRegions):
        regionMap[i] = len(newRegions)
        newRegions.append(regions[i])
    regionList.Region = newRegions
    regionList.RegionCount = len(regionList.Region)
    for data in self.VarData:
        data.VarRegionIndex = [regionMap[i] for i in data.VarRegionIndex]