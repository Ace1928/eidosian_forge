from rdkit.Chem.Pharm2D import SigFactory, Utils
from rdkit.RDLogger import logger
def _ShortestPathsMatch(match, featureSet, sig, dMat, sigFactory):
    """  Internal use only

  """
    if _verbose:
        print('match:', match)
    distsToCheck = Utils.nPointDistDict[len(match)]
    dist = [0] * len(distsToCheck)
    bins = sigFactory.GetBins()
    minD, maxD = (bins[0][0], bins[-1][1])
    for i, (pt0, pt1) in enumerate(distsToCheck):
        minSeen = maxD
        for idx1 in match[pt0]:
            for idx2 in match[pt1]:
                minSeen = min(minSeen, dMat[idx1, idx2])
                if minSeen == 0 or minSeen < minD:
                    return
        d = int(minSeen)
        if d == 0 or d < minD or d >= maxD:
            return None
        dist[i] = d
    idx = sigFactory.GetBitIdx(featureSet, dist, sortIndices=False)
    if _verbose:
        print('\t', dist, minD, maxD, idx)
    if sigFactory.useCounts:
        sig[idx] += 1
    else:
        sig.SetBit(idx)
    return idx