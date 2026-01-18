from rdkit import Chem
from rdkit.Chem import rdmolops
from rdkit.Chem.ChemUtils.DescriptorUtilities import setDescriptorVersion
from collections import defaultdict
def _calculateScoreForAtom(self, atom_idx):
    """Calculates the total score for a single atom in a molecule"""
    atom_score = self.hyb_score[atom_idx] * self.stereo_score[atom_idx] * self.ring_score[atom_idx] * self.bond_score[atom_idx]
    return atom_score