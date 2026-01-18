from rdkit import Chem
from rdkit.Chem import rdmolops
from rdkit.Chem.ChemUtils.DescriptorUtilities import setDescriptorVersion
from collections import defaultdict
def _accountForStereo(self, atom_idx):
    """Calculates the stereo score for a single atom in a molecule"""
    if atom_idx in self.chiral_idxs:
        return 2
    for bond_atom_idxs, stereo in self.doublebonds_stereo.items():
        if stereo != Chem.BondStereo.STEREONONE and atom_idx in bond_atom_idxs:
            return 2
    return 1