import re
from collections import deque, namedtuple
import copy
from numbers import Integral
import numpy as np  # type: ignore
from Bio.PDB.Atom import Atom, DisorderedAtom
from Bio.Data.PDBData import protein_letters_3to1
from Bio.PDB.vectors import multi_coord_space, multi_rot_Z
from Bio.PDB.vectors import coord_space
from Bio.PDB.ic_data import ic_data_backbone, ic_data_sidechains
from Bio.PDB.ic_data import primary_angles
from Bio.PDB.ic_data import ic_data_sidechain_extras, residue_atom_bond_state
from Bio.PDB.ic_data import dihedra_primary_defaults, hedra_defaults
from typing import (
def _add_residue(self, res: 'Residue', last_res: List, last_ord_res: List, verbose: bool=False) -> bool:
    """Set rprev, rnext, manage chain break.

        Returns True for no chain break or residue has sufficient data to
        restart at this position after a chain break (sets initNCaC AtomKeys
        in this case).  False return means insufficient data to extend chain
        with this residue.
        """
    res.internal_coord = IC_Residue(res)
    res.internal_coord.cic = self
    ric = res.internal_coord
    if 0 < len(last_res) and last_ord_res == last_res and (self._peptide_check(last_ord_res[0].residue, res) is None):
        for prev in last_ord_res:
            prev.rnext.append(res.internal_coord)
            ric.rprev.append(prev)
        return True
    elif all((atm in res.child_dict for atm in ('N', 'CA', 'C'))):
        if verbose and len(last_res) != 0:
            if last_ord_res != last_res:
                reason = f'disordered residues after {last_ord_res.pretty_str()}'
            else:
                reason = cast(str, self._peptide_check(last_ord_res[0].residue, res))
            print(f'chain break at {ric.pretty_str()} due to {reason}')
        iNCaC = ric.split_akl((AtomKey(ric, 'N'), AtomKey(ric, 'CA'), AtomKey(ric, 'C')))
        self.initNCaCs.extend(iNCaC)
        return True
    return False