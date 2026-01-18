import warnings
from Bio.Data.PDBData import nucleic_letters_3to1
from Bio.Data.PDBData import nucleic_letters_3to1_extended
from Bio.Data.PDBData import protein_letters_3to1
from Bio.Data.PDBData import protein_letters_3to1_extended
from Bio.PDB.PDBExceptions import PDBException
from Bio.PDB.vectors import calc_dihedral, calc_angle
from Bio.Seq import Seq
class PPBuilder(_PPBuilder):
    """Use C--N distance to find polypeptides."""

    def __init__(self, radius=1.8):
        """Initialize the class."""
        _PPBuilder.__init__(self, radius)

    def _is_connected(self, prev_res, next_res):
        if not prev_res.has_id('C'):
            return False
        if not next_res.has_id('N'):
            return False
        test_dist = self._test_dist
        c = prev_res['C']
        n = next_res['N']
        if c.is_disordered():
            clist = c.disordered_get_list()
        else:
            clist = [c]
        if n.is_disordered():
            nlist = n.disordered_get_list()
        else:
            nlist = [n]
        for nn in nlist:
            for cc in clist:
                n_altloc = nn.get_altloc()
                c_altloc = cc.get_altloc()
                if n_altloc == c_altloc or n_altloc == ' ' or c_altloc == ' ':
                    if test_dist(nn, cc):
                        if c.is_disordered():
                            c.disordered_select(c_altloc)
                        if n.is_disordered():
                            n.disordered_select(n_altloc)
                        return True
        return False

    def _test_dist(self, c, n):
        """Return 1 if distance between atoms<radius (PRIVATE)."""
        if c - n < self.radius:
            return 1
        else:
            return 0