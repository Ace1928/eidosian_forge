import pickle
import numpy
from rdkit.ML.DecTree import CrossValidate, PruneTree
def SetError(self, i, val):
    self.errList[i] = val