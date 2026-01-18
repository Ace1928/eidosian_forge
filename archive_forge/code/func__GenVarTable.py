import numpy
from rdkit.ML.InfoTheory import entropy
def _GenVarTable(vals, cuts, starts, results, nPossibleRes):
    """ Primarily intended for internal use

     constructs a variable table for the data passed in
     The table for a given variable records the number of times each possible value
      of that variable appears for each possible result of the function.

     **Arguments**

       - vals: a 1D Numeric array with the values of the variables

       - cuts: a list with the indices of the quantization bounds
         (indices are into _starts_ )

       - starts: a list of potential starting points for quantization bounds

       - results: a 1D Numeric array of integer result codes

       - nPossibleRes: an integer with the number of possible result codes

     **Returns**

       the varTable, a 2D Numeric array which is nVarValues x nPossibleRes

     **Notes**

       - _vals_ should be sorted!

    """
    nVals = len(cuts) + 1
    varTable = numpy.zeros((nVals, nPossibleRes), 'i')
    idx = 0
    for i in range(nVals - 1):
        cut = cuts[i]
        while idx < starts[cut]:
            varTable[i, results[idx]] += 1
            idx += 1
    while idx < len(vals):
        varTable[-1, results[idx]] += 1
        idx += 1
    return varTable