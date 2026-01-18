from rdkit import RDConfig
from rdkit.ML.Descriptors import Descriptors, Parser
from rdkit.utils import chemutils
def MEAN(self, desc, compos):
    """ *Calculator Method*

      averages the descriptor values across the composition

      **Arguments**

        - desc: the name of the descriptor

        - compos: the composition vector

      **Returns**

        a float

    """
    res = 0.0
    nSoFar = 0.0
    for atom, num in compos:
        res = res + self.atomDict[atom][desc] * num
        nSoFar = nSoFar + num
    return res / nSoFar