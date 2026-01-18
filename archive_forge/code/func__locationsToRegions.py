from fontTools.misc.roundTools import noRound
from .errors import VariationModelError
def _locationsToRegions(self):
    locations = self.locations
    minV = {}
    maxV = {}
    for l in locations:
        for k, v in l.items():
            minV[k] = min(v, minV.get(k, v))
            maxV[k] = max(v, maxV.get(k, v))
    regions = []
    for loc in locations:
        region = {}
        for axis, locV in loc.items():
            if locV > 0:
                region[axis] = (0, locV, maxV[axis])
            else:
                region[axis] = (minV[axis], locV, 0)
        regions.append(region)
    return regions