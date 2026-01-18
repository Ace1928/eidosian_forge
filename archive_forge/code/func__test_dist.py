import warnings
from Bio.Data.PDBData import nucleic_letters_3to1
from Bio.Data.PDBData import nucleic_letters_3to1_extended
from Bio.Data.PDBData import protein_letters_3to1
from Bio.Data.PDBData import protein_letters_3to1_extended
from Bio.PDB.PDBExceptions import PDBException
from Bio.PDB.vectors import calc_dihedral, calc_angle
from Bio.Seq import Seq
def _test_dist(self, c, n):
    """Return 1 if distance between atoms<radius (PRIVATE)."""
    if c - n < self.radius:
        return 1
    else:
        return 0