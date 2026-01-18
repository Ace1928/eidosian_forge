import numpy
from rdkit.ML.InfoTheory import entropy
def FindVarMultQuantBounds(vals, nBounds, results, nPossibleRes):
    """ finds multiple quantization bounds for a single variable

     **Arguments**

       - vals: sequence of variable values (assumed to be floats)

       - nBounds: the number of quantization bounds to find

       - results: a list of result codes (should be integers)

       - nPossibleRes: an integer with the number of possible values of the
         result variable

     **Returns**

       - a 2-tuple containing:

         1) a list of the quantization bounds (floats)

         2) the information gain associated with this quantization


    """
    assert len(vals) == len(results), 'vals/results length mismatch'
    nData = len(vals)
    if nData == 0:
        return ([], -100000000.0)
    svs = list(zip(vals, results))
    svs.sort()
    sortVals, sortResults = zip(*svs)
    startNext = _FindStartPoints(sortVals, sortResults, nData)
    if not len(startNext):
        return ([0], 0.0)
    if len(startNext) < nBounds:
        nBounds = len(startNext) - 1
    if nBounds == 0:
        nBounds = 1
    initCuts = list(range(nBounds))
    maxGain, bestCuts = _RecurseOnBounds(sortVals, initCuts, 0, startNext, sortResults, nPossibleRes)
    quantBounds = []
    nVs = len(sortVals)
    for cut in bestCuts:
        idx = startNext[cut]
        if idx == nVs:
            quantBounds.append(sortVals[-1])
        elif idx == 0:
            quantBounds.append(sortVals[idx])
        else:
            quantBounds.append((sortVals[idx] + sortVals[idx - 1]) / 2.0)
    return (quantBounds, maxGain)