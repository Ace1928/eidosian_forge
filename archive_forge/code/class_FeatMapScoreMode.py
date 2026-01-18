import math
from rdkit.Chem.FeatMaps.FeatMapPoint import FeatMapPoint
class FeatMapScoreMode(object):
    All = 0
    ' score each feature in the probe against every matching\n      feature in the FeatMap.\n  '
    Closest = 1
    ' score each feature in the probe against the closest\n      matching feature in the FeatMap.\n  '
    Best = 2
    ' score each feature in the probe against the matching\n      feature in the FeatMap that leads to the highest score\n  '