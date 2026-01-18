import copy
import math
import numpy
from rdkit import Chem, DataStructs, Geometry
from rdkit.Chem import Draw, rdDepictor
from rdkit.Chem import rdMolDescriptors as rdMD
from rdkit.Chem.Draw import rdMolDraw2D
def GetAPFingerprint(mol, atomId=-1, fpType='normal', nBits=2048, minLength=1, maxLength=30, nBitsPerEntry=4, **kwargs):
    """
    Calculates the atom pairs fingerprint with the torsions of atomId removed.

    Parameters:
      mol -- the molecule of interest
      atomId -- the atom to remove the pairs for (if -1, no pair is removed)
      fpType -- the type of AP fingerprint ('normal', 'hashed', 'bv')
      nBits -- the size of the bit vector (only for fpType='bv')
      minLength -- the minimum path length for an atom pair
      maxLength -- the maxmimum path length for an atom pair
      nBitsPerEntry -- the number of bits available for each pair
    """
    if fpType not in ['normal', 'hashed', 'bv']:
        raise ValueError('Unknown Atom pairs fingerprint type')
    if atomId < 0:
        return apDict[fpType](mol, nBits, minLength, maxLength, nBitsPerEntry, 0, **kwargs)
    if atomId >= mol.GetNumAtoms():
        raise ValueError('atom index greater than number of atoms')
    return apDict[fpType](mol, nBits, minLength, maxLength, nBitsPerEntry, [atomId], **kwargs)