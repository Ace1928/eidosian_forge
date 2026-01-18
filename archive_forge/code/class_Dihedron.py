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
class Dihedron(Edron):
    """Class to represent four joined atoms forming a dihedral angle.

    Attributes
    ----------
    angle: float
        Measurement or specification of dihedral angle in degrees; prefer
        :meth:`IC_Residue.bond_set` to set
    hedron1, hedron2: Hedron object references
        The two hedra which form the dihedral angle
    h1key, h2key: tuples of AtomKeys
        Hash keys for hedron1 and hedron2
    id3,id32: tuples of AtomKeys
        First 3 and second 3 atoms comprising dihedron; hxkey orders may differ
    ric: IC_Residue object reference
        :class:`.IC_Residue` object containing this dihedral
    reverse: bool
        Indicates order of atoms in dihedron is reversed from order of atoms
        in hedra
    primary: bool
        True if this is psi, phi, omega or a sidechain chi angle
    pclass: string (primary angle class)
        re_class with X for adjacent residue according to nomenclature
        (psi, omega, phi)
    cst, rcst: numpy [4][4] arrays
        transformations to (cst) and from (rcst) Dihedron coordinate space
        defined with atom 2 (Hedron 1 center atom) at the origin.  Views on
        :data:`IC_Chain.dCoordSpace`.

    Methods
    -------
    angle()
        getter/setter for dihdral angle in degrees; prefer
        :meth:`IC_Residue.bond_set`
    bits()
        return :data:`IC_Residue.pic_flags` bitmask for dihedron psi, omega, etc
    """

    def __init__(self, *args: Union[List['AtomKey'], DKT], **kwargs: str) -> None:
        """Init Dihedron with sequence of AtomKeys and optional dihedral angle.

        Acceptable input:
            As for Edron, plus optional 'dihedral' keyworded angle value.
        """
        super().__init__(*args, **kwargs)
        self.hedron1: Hedron
        self.hedron2: Hedron
        self.h1key: HKT
        self.h2key: HKT
        self.id3: HKT = cast(HKT, tuple(self.atomkeys[0:3]))
        self.id32: HKT = cast(HKT, tuple(self.atomkeys[1:4]))
        self._setPrimary()
        self.ric: IC_Residue
        self.reverse = False

    def __repr__(self) -> str:
        """Print string for Dihedron object."""
        return f'4-{self.id!s} {self.re_class} {self.angle!s} {self.ric!s}'

    @staticmethod
    def _get_hedron(ic_res: IC_Residue, id3: HKT) -> Optional[Hedron]:
        """Find specified hedron on this residue or its adjacent neighbors."""
        hedron = ic_res.hedra.get(id3, None)
        if not hedron and 0 < len(ic_res.rprev):
            for rp in ic_res.rprev:
                hedron = rp.hedra.get(id3, None)
                if hedron is not None:
                    break
        if not hedron and 0 < len(ic_res.rnext):
            for rn in ic_res.rnext:
                hedron = rn.hedra.get(id3, None)
                if hedron is not None:
                    break
        return hedron

    def _setPrimary(self) -> bool:
        """Mark dihedra required for psi, phi, omega, chi and other angles."""
        dhc = self.e_class
        if dhc == 'NCACN':
            self.pclass = self.re_class[0:7] + 'XN'
            self.primary = True
        elif dhc == 'CACNCA':
            self.pclass = 'XCAXC' + self.re_class[5:]
            self.primary = True
        elif dhc == 'CNCAC':
            self.pclass = 'XC' + self.re_class[2:]
            self.primary = True
        elif dhc == 'CNCACB':
            self.altCB_class = 'XC' + self.re_class[2:]
            self.primary = False
        elif dhc in primary_angles:
            self.primary = True
            self.pclass = self.re_class
        else:
            self.primary = False

    def _set_hedra(self) -> Tuple[bool, Hedron, Hedron]:
        """Work out hedra keys and set rev flag."""
        try:
            return (self.rev, self.hedron1, self.hedron2)
        except AttributeError:
            pass
        rev = False
        res = self.ric
        h1key = self.id3
        hedron1 = Dihedron._get_hedron(res, h1key)
        if not hedron1:
            rev = True
            h1key = cast(HKT, tuple(self.atomkeys[2::-1]))
            hedron1 = Dihedron._get_hedron(res, h1key)
            h2key = cast(HKT, tuple(self.atomkeys[3:0:-1]))
        else:
            h2key = self.id32
        if not hedron1:
            raise HedronMatchError(f"can't find 1st hedron for key {h1key} dihedron {self}")
        hedron2 = Dihedron._get_hedron(res, h2key)
        if not hedron2:
            raise HedronMatchError(f"can't find 2nd hedron for key {h2key} dihedron {self}")
        self.hedron1 = hedron1
        self.h1key = h1key
        self.hedron2 = hedron2
        self.h2key = h2key
        self.reverse = rev
        return (rev, hedron1, hedron2)

    @property
    def angle(self) -> float:
        """Get dihedral angle."""
        try:
            return self.cic.dihedraAngle[self.ndx]
        except AttributeError:
            try:
                return self._dihedral
            except AttributeError:
                return 360.0

    @angle.setter
    def angle(self, dangle_deg_in: float) -> None:
        """Save new dihedral angle; sets needs_update.

        Faster to modify IC_Chain level arrays directly.

        This is probably not the routine you are looking for.  See
        :meth:`IC_Residue.set_angle` or :meth:`IC_Residue.bond_rotate` to change
        a dihedral angle along with its overlapping dihedra, i.e. without
        clashing atoms.

        N.B. dihedron (i-1)C-N-CA-CB is ignored if O exists.
        C-beta is by default placed using O-C-CA-CB, but O is missing
        in some PDB file residues, which means the sidechain cannot be
        placed.  The alternate CB path (i-1)C-N-CA-CB is provided to
        circumvent this, but if this is needed then it must be adjusted in
        conjunction with PHI ((i-1)C-N-CA-C) as they overlap.  This is handled
        by the `IC_Residue` routines above.

        :param float dangle_deg: new dihedral angle in degrees
        """
        if dangle_deg_in > 180.0:
            dangle_deg = dangle_deg_in - 360.0
        elif dangle_deg_in < -180.0:
            dangle_deg = dangle_deg_in + 360.0
        else:
            dangle_deg = dangle_deg_in
        self._dihedral = dangle_deg
        self.needs_update = True
        cic = self.cic
        dndx = self.ndx
        cic.dihedraAngle[dndx] = dangle_deg
        cic.dihedraAngleRads[dndx] = np.deg2rad(dangle_deg)
        cic.dAtoms_needs_update[dndx] = True
        cic.atomArrayValid[cic.atomArrayIndex[self.atomkeys[3]]] = False

    @staticmethod
    def angle_dif(a1: Union[float, np.ndarray], a2: Union[float, np.ndarray]):
        """Get angle difference between two +/- 180 angles.

        https://stackoverflow.com/a/36001014/2783487
        """
        return 180.0 - (180.0 - a2 + a1) % 360.0

    @staticmethod
    def angle_avg(alst: List, in_rads: bool=False, out_rads: bool=False):
        """Get average of list of +/-180 angles.

        :param List alst: list of angles to average
        :param bool in_rads: input values are in radians
        :param bool out_rads: report result in radians
        """
        walst = alst if in_rads else np.deg2rad(alst)
        ravg = np.arctan2(np.sum(np.sin(walst)), np.sum(np.cos(walst)))
        return ravg if out_rads else np.rad2deg(ravg)

    @staticmethod
    def angle_pop_sd(alst: List, avg: float):
        """Get population standard deviation for list of +/-180 angles.

        should be sample std dev but avoid len(alst)=1 -> div by 0
        """
        return np.sqrt(np.sum(np.square(Dihedron.angle_dif(alst, avg))) / len(alst))

    def difference(self, other: 'Dihedron') -> float:
        """Get angle difference between this and other +/- 180 angles."""
        return Dihedron.angle_dif(self.angle, other.angle)

    def bits(self) -> int:
        """Get :data:`IC_Residue.pic_flags` bitmasks for self is psi, omg, phi, pomg, chiX."""
        icr = IC_Residue
        if self.e_class == 'NCACN':
            return icr.pic_flags.psi
        elif hasattr(self, 'pclass') and self.pclass == 'XCAXCPNPCA':
            return icr.pic_flags.omg | icr.pic_flags.pomg
        elif self.e_class == 'CACNCA':
            return icr.pic_flags.omg
        elif self.e_class == 'CNCAC':
            return icr.pic_flags.phi
        else:
            atmNdx = AtomKey.fields.atm
            scList = ic_data_sidechains.get(self.ric.lc)
            aLst = tuple((ak.akl[atmNdx] for ak in self.atomkeys))
            for e in scList:
                if len(e) != 5:
                    continue
                if aLst == e[0:4]:
                    return icr.pic_flags.chi1 << int(e[4][-1]) - 1
        return 0