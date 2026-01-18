from rdkit import RDConfig
from rdkit.ML.Descriptors import Descriptors, Parser
from rdkit.utils import chemutils
def DEV(self, desc, compos):
    """ *Calculator Method*

      average deviation of the descriptor values across the composition

      **Arguments**

        - desc: the name of the descriptor

        - compos: the composition vector

      **Returns**

        a float

    """
    mean = self.MEAN(desc, compos)
    res = 0.0
    nSoFar = 0.0
    for atom, num in compos:
        res = res + abs(self.atomDict[atom][desc] - mean) * num
        nSoFar = nSoFar + num
    return res / nSoFar