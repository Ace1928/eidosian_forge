import math
from rdkit.Chem.FeatMaps.FeatMapPoint import FeatMapPoint
def _loopOverMatchingFeats(self, oFeat):
    for sIdx, sFeat in enumerate(self._feats):
        if sFeat.GetFamily() == oFeat.GetFamily():
            yield (sIdx, sFeat)