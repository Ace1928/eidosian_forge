import pickle
import numpy
from rdkit.ML.DecTree import CrossValidate, PruneTree
def GetDataTuple(self, i):
    """ returns all relevant data about a particular tree in the forest

      **Arguments**

        i: an integer indicating which tree should be returned

      **Returns**

        a 3-tuple consisting of:

          1) the tree

          2) its count

          3) its error
    """
    return (self.treeList[i], self.countList[i], self.errList[i])