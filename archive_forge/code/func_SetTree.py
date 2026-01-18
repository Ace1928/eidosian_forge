import pickle
import numpy
from rdkit.ML.DecTree import CrossValidate, PruneTree
def SetTree(self, i, val):
    self.treeList[i] = val