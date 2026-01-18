import numpy
from rdkit.ML.InfoTheory import entropy
def FindVarQuantBound(vals, results, nPossibleRes):
    """ Uses FindVarMultQuantBounds, only here for historic reasons
    """
    bounds, gain = FindVarMultQuantBounds(vals, 1, results, nPossibleRes)
    return (bounds[0], gain)