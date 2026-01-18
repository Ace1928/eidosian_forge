import pickle
import numpy
from rdkit.ML.DecTree import CrossValidate, PruneTree
def AverageErrors(self):
    """ convert summed error to average error

      This does the conversion in place
    """
    self.errList = [x / y for x, y in zip(self.errList, self.countList)]