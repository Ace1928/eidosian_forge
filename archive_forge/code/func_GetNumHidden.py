import random
import numpy
from rdkit.ML.Neural import ActFuncs, NetNode
def GetNumHidden(self):
    """ returns the number of hidden layers
    """
    return self.numHiddenLayers