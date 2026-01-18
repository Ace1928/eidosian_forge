import pickle
import numpy
from rdkit.ML.DecTree import CrossValidate, PruneTree
def SetCount(self, i, val):
    self.countList[i] = val