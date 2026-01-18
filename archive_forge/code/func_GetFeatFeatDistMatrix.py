import copy
from rdkit.Chem.FeatMaps import FeatMaps
def GetFeatFeatDistMatrix(fm, mergeMetric, mergeTol, dirMergeMode, compatFunc):
    """

    NOTE that mergeTol is a max value for merging when using distance-based
    merging and a min value when using score-based merging.

  """
    MergeMetric.valid(mergeMetric)
    numFeatures = fm.GetNumFeatures()
    dists = [[100000000.0] * numFeatures for _ in range(numFeatures)]
    if mergeMetric == MergeMetric.NoMerge:
        return dists
    benchmarkDict = {MergeMetric.Distance: mergeTol * mergeTol, MergeMetric.Overlap: mergeTol}
    benchmark = benchmarkDict[mergeMetric]

    def assignMatrix(matrix, i, j, value, constraint):
        if value < constraint:
            matrix[i][j] = value
            matrix[j][i] = value
    getFeature = fm.GetFeature
    for i in range(numFeatures):
        ptI = getFeature(i)
        for j in range(i + 1, numFeatures):
            ptJ = getFeature(j)
            if compatFunc(ptI, ptJ):
                if mergeMetric == MergeMetric.Distance:
                    dist2 = ptI.GetDist2(ptJ)
                    assignMatrix(matrix=dists, i=i, j=j, value=dist2, constraint=benchmark)
                elif mergeMetric == MergeMetric.Overlap:
                    score = fm.GetFeatFeatScore(ptI, ptJ, typeMatch=False) * (-1 * ptJ.weight)
                    assignMatrix(matrix=dists, i=i, j=j, value=score, constraint=benchmark)
    return dists