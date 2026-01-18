import warnings
from Bio.Data.PDBData import nucleic_letters_3to1
from Bio.Data.PDBData import nucleic_letters_3to1_extended
from Bio.Data.PDBData import protein_letters_3to1
from Bio.Data.PDBData import protein_letters_3to1_extended
from Bio.PDB.PDBExceptions import PDBException
from Bio.PDB.vectors import calc_dihedral, calc_angle
from Bio.Seq import Seq
class CaPPBuilder(_PPBuilder):
    """Use CA--CA distance to find polypeptides."""

    def __init__(self, radius=4.3):
        """Initialize the class."""
        _PPBuilder.__init__(self, radius)

    def _is_connected(self, prev_res, next_res):
        for r in [prev_res, next_res]:
            if not r.has_id('CA'):
                return False
        n = next_res['CA']
        p = prev_res['CA']
        if n.is_disordered():
            nlist = n.disordered_get_list()
        else:
            nlist = [n]
        if p.is_disordered():
            plist = p.disordered_get_list()
        else:
            plist = [p]
        for nn in nlist:
            for pp in plist:
                if nn - pp < self.radius:
                    return True
        return False