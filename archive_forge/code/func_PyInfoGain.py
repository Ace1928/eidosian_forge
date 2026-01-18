import math
import numpy
def PyInfoGain(varMat):
    """ calculates the information gain for a variable

    **Arguments**

      varMat is a Numeric array with the number of possible occurrences
        of each result for reach possible value of the given variable.

      So, for a variable which adopts 4 possible values and a result which
        has 3 possible values, varMat would be 4x3

    **Returns**

      The expected information gain
  """
    variableRes = numpy.sum(varMat, 1)
    overallRes = numpy.sum(varMat, 0)
    term2 = 0
    for i in range(len(variableRes)):
        term2 = term2 + variableRes[i] * InfoEntropy(varMat[i])
    tSum = sum(overallRes)
    if tSum != 0.0:
        term2 = 1.0 / tSum * term2
        gain = InfoEntropy(overallRes) - term2
    else:
        gain = 0
    return gain