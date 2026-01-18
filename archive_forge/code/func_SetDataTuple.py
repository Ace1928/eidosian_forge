import pickle
import numpy
from rdkit.ML.DecTree import CrossValidate, PruneTree
def SetDataTuple(self, i, tup):
    """ sets all relevant data for a particular tree in the forest

      **Arguments**

        - i: an integer indicating which tree should be returned

        - tup: a 3-tuple consisting of:

          1) the tree

          2) its count

          3) its error
    """
    self.treeList[i], self.countList[i], self.errList[i] = tup