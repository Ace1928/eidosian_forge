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
class Hedron(Edron):
    """Class to represent three joined atoms forming a plane.

    Contains atom coordinates in local coordinate space: central atom
    at origin, one terminal atom on XZ plane, and the other on the +Z axis.
    Stored in two orientations, with the 3rd (forward) or first (reversed)
    atom on the +Z axis.  See :class:`Dihedron` for use of forward and
    reverse orientations.

    Attributes
    ----------
    len12: float
        distance between first and second atoms
    len23: float
        distance between second and third atoms
    angle: float
        angle (degrees) formed by three atoms in hedron
    xrh_class: string
        only for hedron spanning 2 residues, will have 'X' for residue
        contributing only one atom

    Methods
    -------
    get_length()
        get bond length for specified atom pair
    set_length()
        set bond length for specified atom pair
    angle(), len12(), len23()
        setters for relevant attributes (angle in degrees)
    """

    def __init__(self, *args: Union[List['AtomKey'], HKT], **kwargs: str) -> None:
        """Initialize Hedron with sequence of AtomKeys, kwargs.

        Acceptable input:
            As for Edron, plus optional 'len12', 'angle', 'len23'
            keyworded values.
        """
        super().__init__(*args, **kwargs)
        if self.rc == 2:
            resPos = AtomKey.fields.respos
            icode = AtomKey.fields.icode
            resNdx = AtomKey.fields.resname
            atmNdx = AtomKey.fields.atm
            akl0, akl1 = (self.atomkeys[0].akl, self.atomkeys[1].akl)
            if akl0[resPos] != akl1[resPos] or akl0[icode] != akl1[icode]:
                self.xrh_class = 'X' + self.re_class[1:]
            else:
                xrhc = ''
                for i in range(2):
                    xrhc += self.atomkeys[i].akl[resNdx] + self.atomkeys[i].akl[atmNdx]
                self.xrh_class = xrhc + 'X' + self.atomkeys[2].akl[atmNdx]

    def __repr__(self) -> str:
        """Print string for Hedron object."""
        return f'3-{self.id} {self.re_class} {self.len12!s} {self.angle!s} {self.len23!s}'

    @property
    def angle(self) -> float:
        """Get this hedron angle."""
        try:
            return self.cic.hedraAngle[self.ndx]
        except AttributeError:
            return 0.0

    def _invalidate_atoms(self):
        self.cic.hAtoms_needs_update[self.ndx] = True
        for ak in self.atomkeys:
            self.cic.atomArrayValid[self.cic.atomArrayIndex[ak]] = False

    @angle.setter
    def angle(self, angle_deg) -> None:
        """Set this hedron angle; sets needs_update."""
        self.cic.hedraAngle[self.ndx] = angle_deg
        self.cic.hAtoms_needs_update[self.ndx] = True
        self.cic.atomArrayValid[self.cic.atomArrayIndex[self.atomkeys[2]]] = False

    @property
    def len12(self):
        """Get first length for Hedron."""
        try:
            return self.cic.hedraL12[self.ndx]
        except AttributeError:
            return 0.0

    @len12.setter
    def len12(self, len):
        """Set first length for Hedron; sets needs_update."""
        self.cic.hedraL12[self.ndx] = len
        self.cic.hAtoms_needs_update[self.ndx] = True
        self.cic.atomArrayValid[self.cic.atomArrayIndex[self.atomkeys[1]]] = False
        self.cic.atomArrayValid[self.cic.atomArrayIndex[self.atomkeys[2]]] = False

    @property
    def len23(self) -> float:
        """Get second length for Hedron."""
        try:
            return self.cic.hedraL23[self.ndx]
        except AttributeError:
            return 0.0

    @len23.setter
    def len23(self, len):
        """Set second length for Hedron; sets needs_update."""
        self.cic.hedraL23[self.ndx] = len
        self.cic.hAtoms_needs_update[self.ndx] = True
        self.cic.atomArrayValid[self.cic.atomArrayIndex[self.atomkeys[2]]] = False

    def get_length(self, ak_tpl: BKT) -> Optional[float]:
        """Get bond length for specified atom pair.

        :param tuple ak_tpl: tuple of AtomKeys.
            Pair of atoms in this Hedron
        """
        if 2 > len(ak_tpl):
            return None
        if all((ak in self.atomkeys[:2] for ak in ak_tpl)):
            return self.cic.hedraL12[self.ndx]
        if all((ak in self.atomkeys[1:] for ak in ak_tpl)):
            return self.cic.hedraL23[self.ndx]
        return None

    def set_length(self, ak_tpl: BKT, newLength: float):
        """Set bond length for specified atom pair; sets needs_update.

        :param tuple .ak_tpl: tuple of AtomKeys
            Pair of atoms in this Hedron
        """
        if 2 > len(ak_tpl):
            raise TypeError(f'Require exactly 2 AtomKeys: {ak_tpl!s}')
        elif all((ak in self.atomkeys[:2] for ak in ak_tpl)):
            self.cic.hedraL12[self.ndx] = newLength
        elif all((ak in self.atomkeys[1:] for ak in ak_tpl)):
            self.cic.hedraL23[self.ndx] = newLength
        else:
            raise TypeError('%s not found in %s' % (str(ak_tpl), self))
        self._invalidate_atoms()