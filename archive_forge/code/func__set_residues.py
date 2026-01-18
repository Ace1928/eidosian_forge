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
def _set_residues(self, verbose: bool=False) -> None:
    """Initialize .internal_coord for loaded Biopython Residue objects.

        Add IC_Residue as .internal_coord attribute for each :class:`.Residue`
        in parent :class:`Bio.PDB.Chain.Chain`; populate ordered_aa_ic_list with
        :class:`IC_Residue` references for residues which can be built (amino
        acids and some hetatms); set rprev and rnext on each sequential
        IC_Residue, populate initNCaC at start and after chain breaks.

        Generates:
            self.akset : set of :class:`.AtomKey` s in this chain
        """
    last_res: List['IC_Residue'] = []
    last_ord_res: List['IC_Residue'] = []
    akset = set()
    for res in self.chain.get_residues():
        if res.id[0] == ' ' or res.id[0] in IC_Residue.accept_resnames:
            this_res: List['IC_Residue'] = []
            if 2 == res.is_disordered() and (not IC_Residue.no_altloc):
                for r in res.child_dict.values():
                    if self._add_residue(r, last_res, last_ord_res, verbose):
                        this_res.append(r.internal_coord)
                        akset.update(r.internal_coord.ak_set)
            elif self._add_residue(res, last_res, last_ord_res, verbose):
                this_res.append(res.internal_coord)
                akset.update(res.internal_coord.ak_set)
            if 0 < len(this_res):
                self.ordered_aa_ic_list.extend(this_res)
                last_ord_res = this_res
            last_res = this_res
    self.akset = akset
    self.initNCaCs = sorted(self.initNCaCs)