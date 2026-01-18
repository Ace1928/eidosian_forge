import random
from rdkit import Chem
from rdkit.Chem.rdDistGeom import EmbedMolecule
class _StereoGroupFlipper(object):

    def __init__(self, group):
        self._original_parities = [(a, a.GetChiralTag()) for a in group.GetAtoms()]

    def flip(self, flag):
        if flag:
            for a, original_parity in self._original_parities:
                a.SetChiralTag(original_parity)
        else:
            for a, original_parity in self._original_parities:
                if original_parity == Chem.ChiralType.CHI_TETRAHEDRAL_CW:
                    a.SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CCW)
                elif original_parity == Chem.ChiralType.CHI_TETRAHEDRAL_CCW:
                    a.SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CW)