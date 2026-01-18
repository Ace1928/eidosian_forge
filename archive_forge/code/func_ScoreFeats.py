import math
from rdkit.Chem.FeatMaps.FeatMapPoint import FeatMapPoint
def ScoreFeats(self, featsToScore, mapScoreVect=None, featsScoreVect=None, featsToFeatMapIdx=None):
    nFeats = len(self._feats)
    if mapScoreVect is not None:
        if len(mapScoreVect) != nFeats:
            raise ValueError('if provided, len(mapScoreVect) should equal numFeats')
        for i in range(nFeats):
            mapScoreVect[i] = 0.0
    else:
        mapScoreVect = [0.0] * nFeats
    nToScore = len(featsToScore)
    if self.scoreMode == FeatMapScoreMode.Closest:
        defScore = 1000.0
    else:
        defScore = 0.0
    if featsScoreVect is not None:
        if len(featsScoreVect) != nToScore:
            raise ValueError('if provided, len(featsScoreVect) should equal len(featsToScore)')
        for i in range(nToScore):
            featsScoreVect[i] = defScore
    else:
        featsScoreVect = [defScore] * nToScore
    if featsToFeatMapIdx is not None:
        if len(featsToFeatMapIdx) != nToScore:
            raise ValueError('if provided, len(featsToFeatMapIdx) should equal len(featsToScore)')
    else:
        featsToFeatMapIdx = [None] * nToScore
    for i in range(nToScore):
        if self.scoreMode != FeatMapScoreMode.All:
            featsToFeatMapIdx[i] = [-1]
        else:
            featsToFeatMapIdx[i] = []
    for oIdx, oFeat in enumerate(featsToScore):
        for sIdx, sFeat in self._loopOverMatchingFeats(oFeat):
            if self.scoreMode == FeatMapScoreMode.Closest:
                d = sFeat.GetDist2(oFeat)
                if d < featsScoreVect[oIdx]:
                    featsScoreVect[oIdx] = d
                    featsToFeatMapIdx[oIdx][0] = sIdx
            else:
                lScore = self.GetFeatFeatScore(sFeat, oFeat, typeMatch=False)
                if self.scoreMode == FeatMapScoreMode.Best:
                    if lScore > featsScoreVect[oIdx]:
                        featsScoreVect[oIdx] = lScore
                        featsToFeatMapIdx[oIdx][0] = sIdx
                elif self.scoreMode == FeatMapScoreMode.All:
                    featsScoreVect[oIdx] += lScore
                    mapScoreVect[sIdx] += lScore
                    featsToFeatMapIdx[oIdx].append(sIdx)
                else:
                    raise ValueError('bad score mode')
    totScore = 0.0
    if self.scoreMode == FeatMapScoreMode.Closest:
        for oIdx, oFeat in enumerate(featsToScore):
            sIdx = featsToFeatMapIdx[oIdx][0]
            if sIdx > -1:
                lScore = self.GetFeatFeatScore(sFeat, oFeat, typeMatch=False)
                featsScoreVect[oIdx] = lScore
                mapScoreVect[sIdx] = lScore
                totScore += lScore
            else:
                featsScoreVect[oIdx] = 0
    else:
        totScore = sum(featsScoreVect)
        if self.scoreMode == FeatMapScoreMode.Best:
            for oIdx, lScore in enumerate(featsScoreVect):
                sIdx = featsToFeatMapIdx[oIdx][0]
                if sIdx > -1:
                    mapScoreVect[sIdx] = lScore
    if self.scoreMode != FeatMapScoreMode.All:
        for elem in featsToFeatMapIdx:
            if elem == [-1]:
                elem.pop()
    return totScore