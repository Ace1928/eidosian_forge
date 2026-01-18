import pickle
import numpy
from rdkit.ML.DecTree import CrossValidate, PruneTree
def AddTree(self, tree, error):
    """ Adds a tree to the forest

    If an identical tree is already present, its count is incremented

    **Arguments**

      - tree: the new tree

      - error: its error value

    **NOTE:** the errList is run as an accumulator,
        you probably want to call AverageErrors after finishing the forest

    """
    if tree in self.treeList:
        idx = self.treeList.index(tree)
        self.errList[idx] = self.errList[idx] + error
        self.countList[idx] = self.countList[idx] + 1
    else:
        self.treeList.append(tree)
        self.errList.append(error)
        self.countList.append(1)