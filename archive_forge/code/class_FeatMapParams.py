import math
from rdkit.Chem.FeatMaps.FeatMapPoint import FeatMapPoint
class FeatMapParams(object):
    """ one of these should be instantiated for each
  feature type in the feature map
  """
    radius = 2.5
    ' cutoff radius '
    width = 1.0
    ' width parameter (e.g. the gaussian sigma) '

    class FeatProfile(object):
        """ scoring profile of the feature """
        Gaussian = 0
        Triangle = 1
        Box = 2
    featProfile = FeatProfile.Gaussian