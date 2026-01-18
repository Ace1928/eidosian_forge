import numpy
from . import ActFuncs
def SetWeights(self, weights):
    """ Sets the weight list

      **Arguments**

        - weights: a list of values which are to be used as weights

      **Note**

        If this _NetNode_ already has _inputNodes_  and _weights_ is a different length,
        this will bomb out with an assertion.

    """
    if self.inputNodes:
        assert len(weights) == len(self.inputNodes), 'lengths of weights and nodes do not match'
    self.weights = numpy.array(weights)