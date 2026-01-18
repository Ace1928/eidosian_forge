import copy
import numpy
from rdkit.Chem.Pharm2D import Utils
from rdkit.DataStructs import (IntSparseIntVect, LongSparseIntVect,
def GetBitDescriptionAsText(self, bitIdx, includeBins=0, fullPage=1):
    """  returns text with a description of the bit

        **Arguments**

          - bitIdx: an integer bit index

          - includeBins: (optional) if nonzero, information about the bins will be
            included as well

          - fullPage: (optional) if nonzero, html headers and footers will
            be included (so as to make the output a complete page)

        **Returns**

          a string with the HTML

        """
    raise NotImplementedError('Missing implementation')