import numpy
from . import ActFuncs
def Eval(self, valVect):
    """Given a set of inputs (valVect), returns the output of this node

     **Arguments**

      - valVect: a list of inputs

     **Returns**

        the result of running the values in valVect through this node

    """
    if self.inputNodes and len(self.inputNodes) != 0:
        inputs = numpy.take(valVect, self.inputNodes)
        inputs = self.weights * inputs
        val = self.actFunc(sum(inputs))
    else:
        val = 1
    valVect[self.nodeIndex] = val
    return val