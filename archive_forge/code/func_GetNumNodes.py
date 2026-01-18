import random
import numpy
from rdkit.ML.Neural import ActFuncs, NetNode
def GetNumNodes(self):
    """ returns the total number of nodes
    """
    return sum(self.nodeCounts)