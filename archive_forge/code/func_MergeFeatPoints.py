import copy
from rdkit.Chem.FeatMaps import FeatMaps
def MergeFeatPoints(fm, mergeMetric=MergeMetric.NoMerge, mergeTol=1.5, dirMergeMode=DirMergeMode.NoMerge, mergeMethod=MergeMethod.WeightedAverage, compatFunc=familiesMatch):
    """

    NOTE that mergeTol is a max value for merging when using distance-based
    merging and a min value when using score-based merging.

    returns whether or not any points were actually merged

  """
    MergeMetric.valid(mergeMetric)
    MergeMethod.valid(mergeMethod)
    DirMergeMode.valid(dirMergeMode)
    res = False
    if mergeMetric == MergeMetric.NoMerge:
        return res
    dists = GetFeatFeatDistMatrix(fm, mergeMetric, mergeTol, dirMergeMode, compatFunc)
    distOrders = [None] * len(dists)
    for i, distV in enumerate(dists):
        distOrders[i] = []
        for j, dist in enumerate(distV):
            if dist < mergeTol:
                distOrders[i].append((dist, j))
        distOrders[i].sort()
    featsInPlay = list(range(fm.GetNumFeatures()))
    featsToRemove = []
    while featsInPlay:
        fipCopy = featsInPlay[:]
        for fi in fipCopy:
            mergeThem = False
            if not distOrders[fi]:
                featsInPlay.remove(fi)
                continue
            dist, nbr = distOrders[fi][0]
            if nbr not in featsInPlay:
                continue
            if distOrders[nbr][0][1] == fi:
                mergeThem = True
            elif feq(distOrders[nbr][0][0], dist):
                for distJ, nbrJ in distOrders[nbr][1:]:
                    if feq(dist, distJ):
                        if nbrJ == fi:
                            mergeThem = True
                            break
                    else:
                        break
            if mergeThem:
                break
        if mergeThem:
            res = True
            featI = fm.GetFeature(fi)
            nbrFeat = fm.GetFeature(nbr)
            if mergeMethod == MergeMethod.WeightedAverage:
                newPos = featI.GetPos() * featI.weight + nbrFeat.GetPos() * nbrFeat.weight
                newPos /= featI.weight + nbrFeat.weight
                newWeight = (featI.weight + nbrFeat.weight) / 2
            elif mergeMethod == MergeMethod.Average:
                newPos = featI.GetPos() + nbrFeat.GetPos()
                newPos /= 2
                newWeight = (featI.weight + nbrFeat.weight) / 2
            elif mergeMethod == MergeMethod.UseLarger:
                if featI.weight > nbrFeat.weight:
                    newPos = featI.GetPos()
                    newWeight = featI.weight
                else:
                    newPos = nbrFeat.GetPos()
                    newWeight = nbrFeat.weight
            featI.SetPos(newPos)
            featI.weight = newWeight
            featsToRemove.append(nbr)
            featsInPlay.remove(fi)
            featsInPlay.remove(nbr)
            for nbrList in distOrders:
                try:
                    nbrList.remove(fi)
                except ValueError:
                    pass
                try:
                    nbrList.remove(nbr)
                except ValueError:
                    pass
        else:
            break
    featsToRemove.sort()
    for i, fIdx in enumerate(featsToRemove):
        fm.DropFeature(fIdx - i)
    return res