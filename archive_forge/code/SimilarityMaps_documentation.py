import copy
import math
import numpy
from rdkit import Chem, DataStructs, Geometry
from rdkit.Chem import Draw, rdDepictor
from rdkit.Chem import rdMolDescriptors as rdMD
from rdkit.Chem.Draw import rdMolDraw2D

    Calculates the RDKit fingerprint with the paths of atomId removed.

    Parameters:
      mol -- the molecule of interest
      atomId -- the atom to remove the paths for (if -1, no path is removed)
      fpType -- the type of RDKit fingerprint: 'bv'
      nBits -- the size of the bit vector
      minPath -- minimum path length
      maxPath -- maximum path length
      nBitsPerHash -- number of to set per path
    