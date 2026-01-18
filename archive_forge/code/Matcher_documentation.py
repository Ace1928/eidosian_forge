from rdkit import Chem
from rdkit.Chem.Pharm2D import Utils
 Returns a list of lists of atom indices for a bit

    **Arguments**

      - sigFactory: a SigFactory

      - bitIdx: the bit to be queried

      - mol: the molecule to be examined

      - dMat: (optional) the distance matrix of the molecule

      - justOne: (optional) if this is nonzero, only the first match
        will be returned.

      - matchingAtoms: (optional) if this is nonzero, it should
        contain a sequence of sequences with the indices of atoms in
        the molecule which match each of the patterns used by the
        signature.

    **Returns**

      a list of tuples with the matching atoms
  