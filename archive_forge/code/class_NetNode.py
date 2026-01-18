import numpy
from . import ActFuncs
class NetNode:
    """ a node in a neural network

  """

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

    def SetInputs(self, inputNodes):
        """ Sets the input list

      **Arguments**

        - inputNodes: a list of _NetNode_s which are to be used as inputs

      **Note**

        If this _NetNode_ already has weights set and _inputNodes_ is a different length,
        this will bomb out with an assertion.

    """
        if self.weights is not None:
            assert len(self.weights) == len(inputNodes), 'lengths of weights and nodes do not match'
        self.inputNodes = inputNodes[:]

    def GetInputs(self):
        """ returns the input list

    """
        return self.inputNodes

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

    def GetWeights(self):
        """ returns the weight list

    """
        return self.weights

    def __init__(self, nodeIndex, nodeList, inputNodes=None, weights=None, actFunc=ActFuncs.Sigmoid, actFuncParms=()):
        """ Constructor

      **Arguments**

        - nodeIndex: the integer index of this node in _nodeList_

        - nodeList: the list of other _NetNodes_ already in the network

        - inputNodes: a list of this node's inputs

        - weights: a list of this node's weights

        - actFunc: the activation function to be used here.  Must support the API
            of _ActFuncs.ActFunc_.

        - actFuncParms: a tuple of extra arguments to be passed to the activation function
            constructor.

      **Note**
        There should be only one copy of _inputNodes_, every _NetNode_ just has a pointer
        to it so that changes made at one node propagate automatically to the others.

    """
        if inputNodes and weights:
            assert len(weights) == len(inputNodes)
        if weights:
            self.weights = numpy.array(weights)
        else:
            self.weights = None
        if inputNodes:
            self.inputNodes = inputNodes[:]
        else:
            self.inputNodes = None
        self.nodeIndex = nodeIndex
        self.nodeList = nodeList
        self.actFunc = actFunc(*actFuncParms)