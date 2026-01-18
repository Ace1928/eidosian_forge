import numpy
from rdkit.ML.DecTree import DecTree
from rdkit.ML.InfoTheory import entropy
def CalcTotalEntropy(examples, nPossibleVals):
    """ Calculates the total entropy of the data set (w.r.t. the results)

   **Arguments**

    - examples: a list (nInstances long) of lists of variable values + instance
              values
    - nPossibleVals: a list (nVars long) of the number of possible values each variable
      can adopt.

   **Returns**

     a float containing the informational entropy of the data set.

  """
    nRes = nPossibleVals[-1]
    resList = numpy.zeros(nRes, 'i')
    for example in examples:
        res = int(example[-1])
        resList[res] += 1
    return entropy.InfoEntropy(resList)