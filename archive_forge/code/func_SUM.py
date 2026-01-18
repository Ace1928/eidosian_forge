from rdkit import RDConfig
from rdkit.ML.Descriptors import Descriptors, Parser
from rdkit.utils import chemutils
def SUM(self, desc, compos):
    """ *Calculator Method*

      sums the descriptor values across the composition

      **Arguments**

        - desc: the name of the descriptor

        - compos: the composition vector

      **Returns**

        a float

    """
    res = 0.0
    for atom, num in compos:
        res = res + self.atomDict[atom][desc] * num
    return res