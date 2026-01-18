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
def _write_SCAD(self, fp: TextIO, backboneOnly: bool, start=None, fin=None) -> None:
    """Write self to file fp as OpenSCAD data matrices.

        See `OpenSCAD <https://www.openscad.org>`_.
        Works with :func:`.write_SCAD` and embedded OpenSCAD routines therein.
        """
    fp.write(f'   "{self.chain.id}", // chain id\n')
    hedra = {}
    for ric in self.ordered_aa_ic_list:
        respos = ric.residue.id[1]
        if start is not None and respos < start - 1:
            continue
        if fin is not None and respos > fin:
            continue
        for k, h in ric.hedra.items():
            hedra[k] = h
    atomSet: Set[AtomKey] = set()
    bondDict: Dict = {}
    hedraSet: Set[EKT] = set()
    ndx = 0
    hedraNdx = {}
    for hk in sorted(hedra):
        hedraNdx[hk] = ndx
        ndx += 1
    fp.write('   [  // residue array of dihedra')
    resNdx = {}
    dihedraNdx = {}
    ndx = 0
    chnStarted = False
    for ric in self.ordered_aa_ic_list:
        respos = ric.residue.id[1]
        if start is not None and respos < start:
            continue
        if fin is not None and respos > fin:
            continue
        if 'O' not in ric.akc:
            if ric.lc != 'G' and ric.lc != 'A':
                print(f'Unable to generate complete sidechain for {ric} {ric.lc} missing O atom')
        resNdx[ric] = ndx
        if chnStarted:
            fp.write('\n     ],')
        else:
            chnStarted = True
        fp.write('\n     [ // ' + str(ndx) + ' : ' + str(ric.residue.id) + ' ' + ric.lc + ' backbone\n')
        ndx += 1
        ric.clear_transforms()
        ric.assemble(resetLocation=True)
        ndx2 = 0
        started = False
        for i in range(1 if backboneOnly else 2):
            if i == 1:
                cma = ',' if started else ''
                fp.write(f'{cma}\n       // {ric.residue.id!s} {ric.lc} sidechain\n')
            started = False
            for dk, d in sorted(ric.dihedra.items()):
                if d.h2key in hedraNdx and (i == 0 and d.is_backbone() or (i == 1 and (not d.is_backbone()))):
                    if d.cic.dcsValid[d.ndx]:
                        if started:
                            fp.write(',\n')
                        else:
                            started = True
                        fp.write('      ')
                        IC_Chain._writeSCAD_dihed(fp, d, hedraNdx, hedraSet)
                        dihedraNdx[dk] = ndx2
                        hedraSet.add(d.h1key)
                        hedraSet.add(d.h2key)
                        ndx2 += 1
                    else:
                        print(f'Atom missing for {d.id3}-{d.id32}, OpenSCAD chain may be discontiguous')
    fp.write('   ],')
    fp.write('\n  ],\n')
    fp.write('   [  //hedra\n')
    for hk in sorted(hedra):
        hed = hedra[hk]
        fp.write('     [ ')
        fp.write('{:9.5f}, {:9.5f}, {:9.5f}'.format(set_accuracy_95(hed.len12), set_accuracy_95(hed.angle), set_accuracy_95(hed.len23)))
        atom_str = ''
        atom_done_str = ''
        akndx = 0
        for ak in hed.atomkeys:
            atm = ak.akl[AtomKey.fields.atm]
            res = ak.akl[AtomKey.fields.resname]
            ab_state_res = residue_atom_bond_state['X']
            ab_state = ab_state_res.get(atm, None)
            if 'H' == atm[0]:
                ab_state = 'Hsb'
            if ab_state is None:
                ab_state_res = residue_atom_bond_state.get(res, None)
                if ab_state_res is not None:
                    ab_state = ab_state_res.get(atm, '')
                else:
                    ab_state = ''
            atom_str += ', "' + ab_state + '"'
            if ak in atomSet:
                atom_done_str += ', 0'
            elif hk in hedraSet:
                if (hasattr(hed, 'flex_female_1') or hasattr(hed, 'flex_male_1')) and akndx != 2:
                    if akndx == 0:
                        atom_done_str += ', 0'
                    elif akndx == 1:
                        atom_done_str += ', 1'
                        atomSet.add(ak)
                elif (hasattr(hed, 'flex_female_2') or hasattr(hed, 'flex_male_2')) and akndx != 0:
                    if akndx == 2:
                        atom_done_str += ', 0'
                    elif akndx == 1:
                        atom_done_str += ', 1'
                        atomSet.add(ak)
                else:
                    atom_done_str += ', 1'
                    atomSet.add(ak)
            else:
                atom_done_str += ', 0'
            akndx += 1
        fp.write(atom_str)
        fp.write(atom_done_str)
        bond = []
        bond.append(hed.atomkeys[0].id + '-' + hed.atomkeys[1].id)
        bond.append(hed.atomkeys[1].id + '-' + hed.atomkeys[2].id)
        b0 = True
        for b in bond:
            wstr = ''
            if b in bondDict and bondDict[b] == 'StdBond':
                wstr = ', 0'
            elif hk in hedraSet:
                bondType = 'StdBond'
                if b0:
                    if hasattr(hed, 'flex_female_1'):
                        bondType = 'FemaleJoinBond'
                    elif hasattr(hed, 'flex_male_1'):
                        bondType = 'MaleJoinBond'
                    elif hasattr(hed, 'skinny_1'):
                        bondType = 'SkinnyBond'
                    elif hasattr(hed, 'hbond_1'):
                        bondType = 'HBond'
                elif hasattr(hed, 'flex_female_2'):
                    bondType = 'FemaleJoinBond'
                elif hasattr(hed, 'flex_male_2'):
                    bondType = 'MaleJoinBond'
                elif hasattr(hed, 'hbond_2'):
                    bondType = 'HBond'
                if b in bondDict:
                    bondDict[b] = 'StdBond'
                else:
                    bondDict[b] = bondType
                wstr = ', ' + str(bondType)
            else:
                wstr = ', 0'
            fp.write(wstr)
            b0 = False
        akl = hed.atomkeys[0].akl
        fp.write(', "' + akl[AtomKey.fields.resname] + '", ' + akl[AtomKey.fields.respos] + ', "' + hed.e_class + '"')
        fp.write(' ], // ' + str(hk) + '\n')
    fp.write('   ],\n')
    self.atomArrayValid[:] = False
    self.internal_to_atom_coordinates()
    fp.write('\n[  // chain - world transform for each residue\n')
    chnStarted = False
    for ric in self.ordered_aa_ic_list:
        respos = ric.residue.id[1]
        if start is not None and respos < start:
            continue
        if fin is not None and respos > fin:
            continue
        for k, h in ric.hedra.items():
            hedra[k] = h
        for NCaCKey in sorted(ric.NCaCKey):
            mtr = None
            if 0 < len(ric.rprev):
                acl = [self.atomArray[self.atomArrayIndex[ak]] for ak in NCaCKey]
                mt, mtr = coord_space(acl[0], acl[1], acl[2], True)
            else:
                mtr = np.identity(4, dtype=np.float64)
            if chnStarted:
                fp.write(',\n')
            else:
                chnStarted = True
            fp.write('     [ ' + str(resNdx[ric]) + ', "' + str(ric.residue.id[1]))
            fp.write(ric.lc + '", //' + str(NCaCKey) + '\n')
            IC_Chain._write_mtx(fp, mtr)
            fp.write(' ]')
    fp.write('\n   ]\n')