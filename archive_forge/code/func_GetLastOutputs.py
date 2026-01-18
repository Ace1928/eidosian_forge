import random
import numpy
from rdkit.ML.Neural import ActFuncs, NetNode
def GetLastOutputs(self):
    """ returns the complete list of output layer values from the last time this node
    classified anything"""
    return self.lastResults