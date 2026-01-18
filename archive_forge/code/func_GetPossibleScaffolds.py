import itertools
def GetPossibleScaffolds(nPts, bins, useTriangleInequality=True):
    """ gets all realizable scaffolds (passing the triangle inequality) with the
       given number of points and returns them as a list of tuples

    """
    if nPts < 2:
        return 0
    if nPts == 2:
        return [(x,) for x in range(len(bins))]
    nDists = len(nPointDistDict[nPts])
    combos = GetAllCombinations([range(len(bins))] * nDists, noDups=0)
    return [tuple(combo) for combo in combos if not useTriangleInequality or ScaffoldPasses(combo, bins)]