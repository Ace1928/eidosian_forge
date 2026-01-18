import math
from rdkit.Chem.FeatMaps.FeatMapPoint import FeatMapPoint
class FeatProfile(object):
    """ scoring profile of the feature """
    Gaussian = 0
    Triangle = 1
    Box = 2