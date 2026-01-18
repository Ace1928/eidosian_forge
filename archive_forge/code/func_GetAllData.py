import pickle
import numpy
from rdkit.ML.DecTree import CrossValidate, PruneTree
def GetAllData(self):
    """ Returns everything we know

    **Returns**

      a 3-tuple consisting of:

        1) our list of trees

        2) our list of tree counts

        3) our list of tree errors

    """
    return (self.treeList, self.countList, self.errList)