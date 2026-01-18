import math
from rdkit.Chem.FeatMaps.FeatMapPoint import FeatMapPoint
def GetFeatFeatScore(self, feat1, feat2, typeMatch=True):
    """ feat1 is one of our feats
        feat2 is any Feature

    """
    if typeMatch and feat1.GetFamily() != feat2.GetFamily():
        return 0.0
    d2 = feat1.GetDist2(feat2)
    params = self.params[feat1.GetFamily()]
    if d2 > params.radius * params.radius:
        return 0.0
    if params.featProfile == FeatMapParams.FeatProfile.Gaussian:
        score = math.exp(-d2 / params.width)
    elif params.featProfile == FeatMapParams.FeatProfile.Triangle:
        d = math.sqrt(d2)
        if d < params.width:
            score = 1.0 - d / params.width
        else:
            score = 0.0
    elif params.featProfile == FeatMapParams.FeatProfile.Box:
        score = 1.0
    score *= feat1.weight
    if self.dirScoreMode != FeatDirScoreMode.Ignore:
        dirScore = feat1.GetDirMatch(feat2)
        if self.dirScoreMode == FeatDirScoreMode.DotPosRange:
            dirScore = (dirScore + 1.0) / 2.0
        elif self.dirScoreMode != FeatDirScoreMode.DotFullRange:
            raise NotImplementedError('bad feature dir score mode')
        score *= dirScore
    return score