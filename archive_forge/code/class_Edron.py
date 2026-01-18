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
class Edron:
    """Base class for Hedron and Dihedron classes.

    Supports rich comparison based on lists of AtomKeys.

    Attributes
    ----------
    atomkeys: tuple
        3 (hedron) or 4 (dihedron) :class:`.AtomKey` s defining this di/hedron
    id: str
        ':'-joined string of AtomKeys for this di/hedron
    needs_update: bool
        indicates di/hedron local atom_coords do NOT reflect current di/hedron
        angle and length values in hedron local coordinate space
    e_class: str
        sequence of atoms (no position or residue) comprising di/hedron
        for statistics
    re_class: str
        sequence of residue, atoms comprising di/hedron for statistics
    cre_class: str
        sequence of covalent radii classses comprising di/hedron for statistics
    edron_re: compiled regex (Class Attribute)
        A compiled regular expression matching string IDs for Hedron
        and Dihedron objects
    cic: IC_Chain reference
        Chain internal coords object containing this hedron
    ndx: int
        index into IC_Chain level numpy data arrays for di/hedra.
        Set in :meth:`IC_Chain.init_edra`
    rc: int
        number of residues involved in this edron

    Methods
    -------
    gen_key([AtomKey, ...] or AtomKey, ...) (Static Method)
        generate a ':'-joined string of AtomKey Ids
    is_backbone()
        Return True if all atomkeys atoms are N, Ca, C or O

    """
    edron_re = re.compile('^(?P<pdbid>\\w+)?\\s(?P<chn>[\\w|\\s])?\\s(?P<a1>[\\w\\-\\.]+):(?P<a2>[\\w\\-\\.]+):(?P<a3>[\\w\\-\\.]+)(:(?P<a4>[\\w\\-\\.]+))?\\s+(((?P<len12>\\S+)\\s+(?P<angle>\\S+)\\s+(?P<len23>\\S+)\\s*$)|((?P<dihedral>\\S+)\\s*$))')
    ' A compiled regular expression matching string IDs for Hedron and\n    Dihedron objects'

    @staticmethod
    def gen_key(lst: List['AtomKey']) -> str:
        """Generate string of ':'-joined AtomKey strings from input.

        Generate '2_A_C:3_P_N:3_P_CA' from (2_A_C, 3_P_N, 3_P_CA)
        :param list lst: list of AtomKey objects
        """
        if 4 == len(lst):
            return f'{lst[0].id}:{lst[1].id}:{lst[2].id}:{lst[3].id}'
        else:
            return f'{lst[0].id}:{lst[1].id}:{lst[2].id}'

    @staticmethod
    def gen_tuple(akstr: str) -> Tuple:
        """Generate AtomKey tuple for ':'-joined AtomKey string.

        Generate (2_A_C, 3_P_N, 3_P_CA) from '2_A_C:3_P_N:3_P_CA'
        :param str akstr: string of ':'-separated AtomKey strings
        """
        return tuple([AtomKey(i) for i in akstr.split(':')])

    def __init__(self, *args: Union[List['AtomKey'], EKT], **kwargs: str) -> None:
        """Initialize Edron with sequence of AtomKeys.

        Acceptable input:

            [ AtomKey, ... ]  : list of AtomKeys
            AtomKey, ...      : sequence of AtomKeys as args
            {'a1': str, 'a2': str, ... }  : dict of AtomKeys as 'a1', 'a2' ...
        """
        atomkeys: List[AtomKey] = []
        for arg in args:
            if isinstance(arg, list):
                atomkeys = arg
            elif isinstance(arg, tuple):
                atomkeys = list(arg)
            elif arg is not None:
                atomkeys.append(arg)
        if [] == atomkeys and all((k in kwargs for k in ('a1', 'a2', 'a3'))):
            atomkeys = [AtomKey(kwargs['a1']), AtomKey(kwargs['a2']), AtomKey(kwargs['a3'])]
            if 'a4' in kwargs and kwargs['a4'] is not None:
                atomkeys.append(AtomKey(kwargs['a4']))
        self.atomkeys = tuple(atomkeys)
        self.id = Edron.gen_key(atomkeys)
        self._hash = hash(self.atomkeys)
        self.needs_update = True
        self.cic: IC_Chain
        self.e_class = ''
        self.re_class = ''
        self.cre_class = ''
        rset = set()
        atmNdx = AtomKey.fields.atm
        resNdx = AtomKey.fields.resname
        resPos = AtomKey.fields.respos
        icode = AtomKey.fields.icode
        for ak in atomkeys:
            akl = ak.akl
            self.e_class += akl[atmNdx]
            self.re_class += akl[resNdx] + akl[atmNdx]
            rset.add(akl[resPos] + (akl[icode] or ''))
            self.cre_class += ak.cr_class()
        self.rc = len(rset)

    def __deepcopy__(self, memo):
        """Deep copy implementation for Edron."""
        existing = memo.get(id(self), False)
        if existing:
            return existing
        dup = type(self).__new__(self.__class__)
        memo[id(self)] = dup
        dup.__dict__.update(self.__dict__)
        dup.cic = memo[id(self.cic)]
        dup.atomkeys = copy.deepcopy(self.atomkeys, memo)
        return dup

    def __contains__(self, ak: 'AtomKey') -> bool:
        """Return True if atomkey is in this edron."""
        return ak in self.atomkeys

    def is_backbone(self) -> bool:
        """Report True for contains only N, C, CA, O, H atoms."""
        return all((ak.is_backbone() for ak in self.atomkeys))

    def __repr__(self) -> str:
        """Tuple of AtomKeys is default repr string."""
        return str(self.atomkeys)

    def __hash__(self) -> int:
        """Hash calculated at init from atomkeys tuple."""
        return self._hash

    def _cmp(self, other: 'Edron') -> Union[Tuple['AtomKey', 'AtomKey'], bool]:
        """Comparison function ranking self vs. other; False on equal.

        Priority is lowest value for sort: psi < chi1.
        """
        for ak_s, ak_o in zip(self.atomkeys, other.atomkeys):
            if ak_s != ak_o:
                return (ak_s, ak_o)
        return False

    def __eq__(self, other: object) -> bool:
        """Test for equality."""
        if not isinstance(other, type(self)):
            return NotImplemented
        return self.id == other.id

    def __ne__(self, other: object) -> bool:
        """Test for inequality."""
        if not isinstance(other, type(self)):
            return NotImplemented
        return self.id != other.id

    def __gt__(self, other: object) -> bool:
        """Test greater than."""
        if not isinstance(other, type(self)):
            return NotImplemented
        rslt = self._cmp(other)
        if rslt:
            rslt = cast(Tuple[AtomKey, AtomKey], rslt)
            return rslt[0] > rslt[1]
        return False

    def __ge__(self, other: object) -> bool:
        """Test greater or equal."""
        if not isinstance(other, type(self)):
            return NotImplemented
        rslt = self._cmp(other)
        if rslt:
            rslt = cast(Tuple[AtomKey, AtomKey], rslt)
            return rslt[0] >= rslt[1]
        return True

    def __lt__(self, other: object) -> bool:
        """Test less than."""
        if not isinstance(other, type(self)):
            return NotImplemented
        rslt = self._cmp(other)
        if rslt:
            rslt = cast(Tuple[AtomKey, AtomKey], rslt)
            return rslt[0] < rslt[1]
        return False

    def __le__(self, other: object) -> bool:
        """Test less or equal."""
        if not isinstance(other, type(self)):
            return NotImplemented
        rslt = self._cmp(other)
        if rslt:
            rslt = cast(Tuple[AtomKey, AtomKey], rslt)
            return rslt[0] <= rslt[1]
        return True