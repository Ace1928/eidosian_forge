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
def _hedraDict2chain(self, hl12: Dict[str, float], ha: Dict[str, float], hl23: Dict[str, float], da: Dict[str, float], bfacs: Dict[str, float]) -> None:
    """Generate chain numpy arrays from :func:`.read_PIC` dicts.

        On entry:
            * chain internal_coord has ordered_aa_ic_list built, akset;
            * residues have rnext, rprev, ak_set and di/hedra dicts initialised
            * Chain, residues do NOT have NCaC info, id3_dh_index
            * Di/hedra have cic, atomkeys set
            * Dihedra do NOT have valid reverse flag, h1/2 info

        """
    for ric in self.ordered_aa_ic_list:
        initNCaC = []
        for atm in ric.residue.get_atoms():
            if 2 == atm.is_disordered():
                if IC_Residue.no_altloc:
                    initNCaC.append(AtomKey(ric, atm.selected_child))
                else:
                    for altAtom in atm.child_dict.values():
                        if altAtom.coord is not None:
                            initNCaC.append(AtomKey(ric, altAtom))
            elif atm.coord is not None:
                initNCaC.append(AtomKey(ric, atm))
        if initNCaC != []:
            self.initNCaCs.append(tuple(initNCaC))
        ric.NCaCKey = []
        ric.NCaCKey.extend(ric.split_akl((AtomKey(ric, 'N'), AtomKey(ric, 'CA'), AtomKey(ric, 'C'))))
        ric._link_dihedra()
    if self.initNCaCs == []:
        ric = self.ordered_aa_ic_list[0]
        iNCaC = ric.split_akl((AtomKey(ric, 'N'), AtomKey(ric, 'CA'), AtomKey(ric, 'C')))
        self.initNCaCs.extend(iNCaC)
    self.build_atomArray()
    self.initNCaCs = sorted(self.initNCaCs)
    spNdx, icNdx, resnNdx, atmNdx, altlocNdx, occNdx = AtomKey.fields
    sn = None
    for ak, ndx in self.atomArrayIndex.items():
        res = ak.ric.residue
        atm, altloc = (ak.akl[atmNdx], ak.akl[altlocNdx])
        occ = 1.0 if ak.akl[occNdx] is None else float(ak.akl[occNdx])
        bfac = bfacs.get(ak.id, 0.0)
        sn = sn + 1 if sn is not None else ndx + 1
        bpAtm = None
        if res.has_id(atm):
            bpAtm = res[atm]
        if bpAtm is None or (2 == bpAtm.is_disordered() and (not bpAtm.disordered_has_id(altloc))):
            newAtom = Atom(atm, self.atomArray[ndx][0:3], bfac, occ, ' ' if altloc is None else altloc, atm, sn, atm[0])
            if bpAtm is None:
                if altloc is None:
                    res.add(newAtom)
                else:
                    disordered_atom = DisorderedAtom(atm)
                    res.add(disordered_atom)
                    disordered_atom.disordered_add(newAtom)
                    res.flag_disordered()
            else:
                bpAtm.disordered_add(newAtom)
        else:
            if 2 == bpAtm.is_disordered() and bpAtm.disordered_has_id(altloc):
                bpAtm.disordered_select(altloc)
            bpAtm.set_bfactor(bfac)
            bpAtm.set_occupancy(occ)
            sn = bpAtm.get_serial_number()
    self.hedraLen = len(ha)
    self.hedraL12 = np.fromiter(hl12.values(), dtype=np.float64)
    self.hedraAngle = np.fromiter(ha.values(), dtype=np.float64)
    self.hedraL23 = np.fromiter(hl23.values(), dtype=np.float64)
    self.hedraNdx = dict(zip(sorted(ha.keys()), range(self.hedraLen)))
    self.dihedraLen = len(da)
    self.dihedraAngle = np.fromiter(da.values(), dtype=np.float64)
    self.dihedraAngleRads = np.deg2rad(self.dihedraAngle)
    self.dihedraNdx = dict(zip(sorted(da.keys()), range(self.dihedraLen)))
    self.build_edraArrays()