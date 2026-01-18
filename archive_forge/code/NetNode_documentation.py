import numpy
from . import ActFuncs
 Constructor

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

    