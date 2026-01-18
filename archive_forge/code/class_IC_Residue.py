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
class IC_Residue:
    """Class to extend Biopython Residue with internal coordinate data.

    Parameters
    ----------
    parent: biopython Residue object this class extends

    Attributes
    ----------
    no_altloc: bool default False
        **Class** variable, disable processing of ALTLOC atoms if True, use
        only selected atoms.

    accept_atoms: tuple
        **Class** variable :data:`accept_atoms`, list of PDB atom names to use
        when generating internal coordinates.
        Default is::

            accept_atoms = accept_mainchain + accept_hydrogens

        to exclude hydrogens in internal coordinates and generated PDB files,
        override as::

            IC_Residue.accept_atoms = IC_Residue.accept_mainchain

        to get only mainchain atoms plus amide proton, use::

            IC_Residue.accept_atoms = IC_Residue.accept_mainchain + ('H',)

        to convert D atoms to H, set :data:`AtomKey.d2h` = True and use::

            IC_Residue.accept_atoms = (
                accept_mainchain + accept_hydrogens + accept_deuteriums
            )

        Note that `accept_mainchain = accept_backbone + accept_sidechain`.
        Thus to generate sequence-agnostic conformational data for e.g.
        structure alignment in dihedral angle space, use::

            IC_Residue.accept_atoms = accept_backbone

        or set gly_Cbeta = True and use::

            IC_Residue.accept_atoms = accept_backbone + ('CB',)

        Changing accept_atoms will cause the default `structure_rebuild_test` in
        :mod:`.ic_rebuild` to fail if some atoms are filtered (obviously).  Use
        the `quick=True` option to test only the coordinates of filtered atoms
        to avoid this.

        There is currently no option to output internal coordinates with D
        instead of H.

    accept_resnames: tuple
        **Class** variable :data:`accept_resnames`, list of 3-letter residue
        names for HETATMs to accept when generating internal coordinates from
        atoms.  HETATM sidechain will be ignored, but normal backbone atoms (N,
        CA, C, O, CB) will be included.  Currently only CYG, YCM and UNK;
        override at your own risk.  To generate sidechain, add appropriate
        entries to `ic_data_sidechains` in :mod:`.ic_data` and support in
        :meth:`IC_Chain.atom_to_internal_coordinates`.

    gly_Cbeta: bool default False
        **Class** variable :data:`gly_Cbeta`, override to True to generate
        internal coordinates for glycine CB atoms in
        :meth:`IC_Chain.atom_to_internal_coordinates` ::

            IC_Residue.gly_Cbeta = True

    pic_accuracy: str default "17.13f"
        **Class** variable :data:`pic_accuracy` sets accuracy for numeric values
        (angles, lengths) in .pic files.  Default set high to support mmCIF file
        accuracy in rebuild tests.  If you find rebuild tests fail with
        'ERROR -COORDINATES-' and verbose=True shows only small discrepancies,
        try raising this value (or lower it to 9.5 if only working with PDB
        format files).  ::

            IC_Residue.pic_accuracy = "9.5f"

    residue: Biopython Residue object reference
        The :class:`.Residue` object this extends
    hedra: dict indexed by 3-tuples of AtomKeys
        Hedra forming this residue
    dihedra: dict indexed by 4-tuples of AtomKeys
        Dihedra forming (overlapping) this residue
    rprev, rnext: lists of IC_Residue objects
        References to adjacent (bonded, not missing, possibly disordered)
        residues in chain
    atom_coords: AtomKey indexed dict of numpy [4] arrays
        **removed**
        Use AtomKeys and atomArrayIndex to build if needed
    ak_set: set of AtomKeys in dihedra
        AtomKeys in all dihedra overlapping this residue (see __contains__())
    alt_ids: list of char
        AltLoc IDs from PDB file
    bfactors: dict
        AtomKey indexed B-factors as read from PDB file
    NCaCKey: List of tuples of AtomKeys
        List of tuples of N, Ca, C backbone atom AtomKeys; usually only 1
        but more if backbone altlocs.
    is20AA: bool
        True if residue is one of 20 standard amino acids, based on
        Residue resname
    isAccept: bool
        True if is20AA or in accept_resnames below
    rbase: tuple
        residue position, insert code or none, resname (1 letter if standard
        amino acid)
    cic: IC_Chain default None
        parent chain :class:`IC_Chain` object
    scale: optional float
        used for OpenSCAD output to generate gly_Cbeta bond length

    Methods
    -------
    assemble(atomCoordsIn, resetLocation, verbose)
        Compute atom coordinates for this residue from internal coordinates
    get_angle()
        Return angle for passed key
    get_length()
        Return bond length for specified pair
    pick_angle()
        Find Hedron or Dihedron for passed key
    pick_length()
        Find hedra for passed AtomKey pair
    set_angle()
        Set angle for passed key (no position updates)
    set_length()
        Set bond length in all relevant hedra for specified pair
    bond_rotate(delta)
        adjusts related dihedra angles by delta, e.g. rotating psi (N-Ca-C-N)
        will adjust the adjacent N-Ca-C-O by the same amount to avoid clashes
    bond_set(angle)
        uses bond_rotate to set specified dihedral to angle and adjust related
        dihedra accordingly
    rak(atom info)
        cached AtomKeys for this residue
    """
    accept_resnames = ('CYG', 'YCM', 'UNK')
    'Add 3-letter residue name here for non-standard residues with\n    normal backbone.  CYG included for test case 4LGY (1305 residue\n    contiguous chain).  Safe to add more names for N-CA-C-O backbones, any\n    more complexity will need additions to :data:`accept_atoms`,\n    `ic_data_sidechains` in :mod:`.ic_data` and support in\n    :meth:`IC_Chain.atom_to_internal_coordinates`'
    _AllBonds: bool = False
    'For OpenSCAD output, generate explicit hedra covering all bonds.\n    **Class** variable, whereas a PDB file just specifies atoms, OpenSCAD\n    output for 3D printing needs all bonds specified explicitly - otherwise\n    e.g. PHE rings will not be closed.  This variable is managed by the\n    :func:`.SCADIO.write_SCAD` code.'
    no_altloc: bool = False
    'Set True to filter altloc atoms on input and only work with Biopython\n    default Atoms'
    gly_Cbeta: bool = False
    "Create beta carbons on all Gly residues.\n\n    Setting this to True will generate internal coordinates for Gly C-beta\n    carbons in :meth:`atom_to_internal_coordinates`.\n\n    Data averaged from Sep 2019 Dunbrack cullpdb_pc20_res2.2_R1.0\n    restricted to structures with amide protons.\n    Please see\n\n    `PISCES: A Protein Sequence Culling Server <https://dunbrack.fccc.edu/pisces/>`_\n\n    'G. Wang and R. L. Dunbrack, Jr. PISCES: a protein sequence culling\n    server. Bioinformatics, 19:1589-1591, 2003.'\n\n    Ala avg rotation of OCCACB from NCACO query::\n\n        select avg(g.rslt) as avg_rslt, stddev(g.rslt) as sd_rslt, count(*)\n        from\n        (select f.d1d, f.d2d,\n        (case when f.rslt > 0 then f.rslt-360.0 else f.rslt end) as rslt\n        from (select d1.angle as d1d, d2.angle as d2d,\n        (d2.angle - d1.angle) as rslt from dihedron d1,\n        dihedron d2 where d1.re_class='AOACACAACB' and\n        d2.re_class='ANACAACAO' and d1.pdb=d2.pdb and d1.chn=d2.chn\n        and d1.res=d2.res) as f) as g\n\n    results::\n\n        | avg_rslt          | sd_rslt          | count   |\n        | -122.682194862932 | 5.04403040513919 | 14098   |\n"
    pic_accuracy: str = '17.13f'
    accept_backbone = ('N', 'CA', 'C', 'O', 'OXT')
    accept_sidechain = ('CB', 'CG', 'CG1', 'OG1', 'OG', 'SG', 'CG2', 'CD', 'CD1', 'SD', 'OD1', 'ND1', 'CD2', 'ND2', 'CE', 'CE1', 'NE', 'OE1', 'NE1', 'CE2', 'OE2', 'NE2', 'CE3', 'CZ', 'NZ', 'CZ2', 'CZ3', 'OD2', 'OH', 'CH2', 'NH1', 'NH2')
    accept_mainchain = accept_backbone + accept_sidechain
    accept_hydrogens = ('H', 'H1', 'H2', 'H3', 'HA', 'HA2', 'HA3', 'HB', 'HB1', 'HB2', 'HB3', 'HG2', 'HG3', 'HD2', 'HD3', 'HE2', 'HE3', 'HZ1', 'HZ2', 'HZ3', 'HG11', 'HG12', 'HG13', 'HG21', 'HG22', 'HG23', 'HZ', 'HD1', 'HE1', 'HD11', 'HD12', 'HD13', 'HG', 'HG1', 'HD21', 'HD22', 'HD23', 'NH1', 'NH2', 'HE', 'HH11', 'HH12', 'HH21', 'HH22', 'HE21', 'HE22', 'HE2', 'HH', 'HH2')
    accept_deuteriums = ('D', 'D1', 'D2', 'D3', 'DA', 'DA2', 'DA3', 'DB', 'DB1', 'DB2', 'DB3', 'DG2', 'DG3', 'DD2', 'DD3', 'DE2', 'DE3', 'DZ1', 'DZ2', 'DZ3', 'DG11', 'DG12', 'DG13', 'DG21', 'DG22', 'DG23', 'DZ', 'DD1', 'DE1', 'DD11', 'DD12', 'DD13', 'DG', 'DG1', 'DD21', 'DD22', 'DD23', 'ND1', 'ND2', 'DE', 'DH11', 'DH12', 'DH21', 'DH22', 'DE21', 'DE22', 'DE2', 'DH', 'DH2')
    accept_atoms = accept_mainchain + accept_hydrogens
    'Change accept_atoms to restrict atoms processed. See :class:`IC_Residue`\n    for usage.'

    def __init__(self, parent: 'Residue') -> None:
        """Initialize IC_Residue with parent Biopython Residue.

        :param Residue parent: Biopython Residue object.
            The Biopython Residue this object extends
        """
        self.residue = parent
        self.cic: IC_Chain
        self.hedra: Dict[HKT, Hedron] = {}
        self.dihedra: Dict[DKT, Dihedron] = {}
        self.akc: Dict[Union[str, Atom], AtomKey] = {}
        self.ak_set: Set[AtomKey] = set()
        self.rprev: List[IC_Residue] = []
        self.rnext: List[IC_Residue] = []
        self.bfactors: Dict[str, float] = {}
        self.alt_ids: Union[List[str], None] = None if IC_Residue.no_altloc else []
        self.is20AA = True
        self.isAccept = True
        rid = parent.id
        rbase = [rid[1], rid[2] if ' ' != rid[2] else None, parent.resname]
        try:
            rbase[2] = protein_letters_3to1[rbase[2]]
        except KeyError:
            self.is20AA = False
            if rbase[2] not in self.accept_resnames:
                self.isAccept = False
        self.rbase = tuple(rbase)
        self.lc = rbase[2]
        if self.isAccept:
            for atom in parent.get_atoms():
                if hasattr(atom, 'child_dict'):
                    if IC_Residue.no_altloc:
                        self._add_atom(atom.selected_child)
                    else:
                        for atm in atom.child_dict.values():
                            self._add_atom(atm)
                else:
                    self._add_atom(atom)
            if self.ak_set:
                self._build_rak_cache()

    def __deepcopy__(self, memo):
        """Deep copy implementation for IC_Residue."""
        existing = memo.get(id(self), False)
        if existing:
            return existing
        dup = type(self).__new__(self.__class__)
        memo[id(self)] = dup
        dup.__dict__.update(self.__dict__)
        dup.cic = memo[id(self.cic)]
        dup.residue = memo[id(self.residue)]
        return dup

    def __contains__(self, ak: 'AtomKey') -> bool:
        """Return True if atomkey is in this residue."""
        if ak in self.ak_set:
            akl = ak.akl
            if int(akl[0]) == self.rbase[0] and akl[1] == self.rbase[1] and (akl[2] == self.rbase[2]):
                return True
        return False

    def rak(self, atm: Union[str, Atom]) -> 'AtomKey':
        """Cache calls to AtomKey for this residue."""
        try:
            ak = self.akc[atm]
        except KeyError:
            ak = self.akc[atm] = AtomKey(self, atm)
            if isinstance(atm, str):
                ak.missing = True
        return ak

    def _build_rak_cache(self) -> None:
        """Create explicit entries for for atoms so don't miss altlocs.

        This ensures that self.akc (atom key cache) has an entry for selected
        atom name (e.g. "CA") amongst any that have altlocs.  Without this,
        rak() on the other altloc atom first may result in the main atom being
        missed.
        """
        for ak in sorted(self.ak_set):
            atmName = ak.akl[3]
            if self.akc.get(atmName) is None:
                self.akc[atmName] = ak

    def _add_atom(self, atm: Atom) -> None:
        """Filter Biopython Atom with accept_atoms; set ak_set.

        Arbitrarily renames O' and O'' to O and OXT
        """
        if 'O' == atm.name[0]:
            if "O'" == atm.name:
                atm.name = 'O'
            elif "O''" == atm.name:
                atm.name = 'OXT'
        if atm.name not in self.accept_atoms:
            return
        ak = self.rak(atm)
        self.ak_set.add(ak)

    def __repr__(self) -> str:
        """Print string is parent Residue ID."""
        return str(self.residue.full_id)

    def pretty_str(self) -> str:
        """Nice string for residue ID."""
        id = self.residue.id
        return f'{self.residue.resname} {id[0]}{id[1]!s}{id[2]}'

    def _link_dihedra(self, verbose: bool=False) -> None:
        """Housekeeping after loading all residues and dihedra.

        - Link dihedra to this residue
        - form id3_dh_index
        - form ak_set
        - set NCaCKey to be available AtomKeys

        called for loading PDB / atom coords
        """
        for dh in self.dihedra.values():
            dh.ric = self
            dh.cic = self.cic
            self.ak_set.update(dh.atomkeys)
        for h in self.hedra.values():
            self.ak_set.update(h.atomkeys)
            h.cic = self.cic
        if not self.akc:
            self._build_rak_cache()
        self.NCaCKey = []
        self.NCaCKey.extend(self.split_akl((AtomKey(self, 'N'), AtomKey(self, 'CA'), AtomKey(self, 'C'))))

    def set_flexible(self) -> None:
        """For OpenSCAD, mark N-CA and CA-C bonds to be flexible joints.

        See :func:`.SCADIO.write_SCAD`
        """
        for h in self.hedra.values():
            if h.e_class == 'NCAC':
                h.flex_female_1 = True
                h.flex_female_2 = True
            elif h.e_class.endswith('NCA'):
                h.flex_male_2 = True
            elif h.e_class.startswith('CAC') and h.atomkeys[1].akl[3] == 'C':
                h.flex_male_1 = True
            elif h.e_class == 'CBCAC':
                h.skinny_1 = True

    def set_hbond(self) -> None:
        """For OpenSCAD, mark H-N and C-O bonds to be hbonds (magnets).

        See :func:`.SCADIO.write_SCAD`
        """
        for h in self.hedra.values():
            if h.e_class == 'HNCA':
                h.hbond_1 = True
            elif h.e_class == 'CACO':
                h.hbond_2 = True

    def _default_startpos(self) -> Dict['AtomKey', np.array]:
        """Generate default N-Ca-C coordinates to build this residue from."""
        atomCoords = {}
        cic = self.cic
        dlist0 = [cic.id3_dh_index.get(akl, None) for akl in sorted(self.NCaCKey)]
        dlist1 = [d for d in dlist0 if d is not None]
        dlist = [cic.dihedra[val] for sublist in dlist1 for val in sublist]
        for d in dlist:
            for i, a in enumerate(d.atomkeys):
                atomCoords[a] = cic.dAtoms[d.ndx][i]
        return atomCoords

    def _get_startpos(self) -> Dict['AtomKey', np.array]:
        """Find N-Ca-C coordinates to build this residue from."""
        startPos = {}
        cic = self.cic
        for ncac in self.NCaCKey:
            if np.all(cic.atomArrayValid[[cic.atomArrayIndex[ak] for ak in ncac]]):
                for ak in ncac:
                    startPos[ak] = cic.atomArray[cic.atomArrayIndex[ak]]
        if startPos == {}:
            startPos = self._default_startpos()
        return startPos

    def clear_transforms(self):
        """Invalidate dihedra coordinate space attributes before assemble().

        Coordinate space attributes are Dihedron.cst and .rcst, and
        :data:`IC_Chain.dCoordSpace`
        """
        for d in self.dihedra.values():
            self.cic.dcsValid[d.ndx] = False

    def assemble(self, resetLocation: bool=False, verbose: bool=False) -> Union[Dict['AtomKey', np.array], Dict[HKT, np.array], None]:
        """Compute atom coordinates for this residue from internal coordinates.

        This is the IC_Residue part of the :meth:`.assemble_residues_ser` serial
        version, see :meth:`.assemble_residues` for numpy vectorized approach
        which works at the :class:`IC_Chain` level.

        Join prepared dihedra starting from N-CA-C and N-CA-CB hedrons,
        computing protein space coordinates for backbone and sidechain atoms

        Sets forward and reverse transforms on each Dihedron to convert from
        protein coordinates to dihedron space coordinates for first three
        atoms (see :data:`IC_Chain.dCoordSpace`)

        Call :meth:`.init_atom_coords` to update any modified di/hedra before
        coming here, this only assembles dihedra into protein coordinate space.

        **Algorithm**

        Form double-ended queue, start with c-ca-n, o-c-ca, n-ca-cb, n-ca-c.

        if resetLocation=True, use initial coords from generating dihedron
        for n-ca-c initial positions (result in dihedron coordinate space)

        while queue not empty
            get 3-atom hedron key

            for each dihedron starting with hedron key (1st hedron of dihedron)

                if have coordinates for all 4 atoms already
                    add 2nd hedron key to back of queue
                else if have coordinates for 1st 3 atoms
                    compute forward and reverse transforms to take 1st 3 atoms
                    to/from dihedron initial coordinate space

                    use reverse transform to get position of 4th atom in
                    current coordinates from dihedron initial coordinates

                    add 2nd hedron key to back of queue
                else
                    ordering failed, put hedron key at back of queue and hope
                    next time we have 1st 3 atom positions (should not happen)

        loop terminates (queue drains) as hedron keys which do not start any
        dihedra are removed without action

        :param bool resetLocation: default False.
            - Option to ignore start location and orient so initial N-Ca-C
            hedron at origin.

        :returns:
            Dict of AtomKey -> homogeneous atom coords for residue in protein
            space relative to previous residue

            **Also** directly updates :data:`IC_Chain.atomArray` as
            :meth:`.assemble_residues` does.

        """
        cic = self.cic
        dcsValid = cic.dcsValid
        aaValid = cic.atomArrayValid
        aaNdx = cic.atomArrayIndex
        aa = cic.atomArray
        if not self.ak_set:
            return None
        NCaCKey = sorted(self.NCaCKey)
        rseqpos = self.rbase[0]
        startLst = self.split_akl((self.rak('C'), self.rak('CA'), self.rak('N')))
        if 'CB' in self.akc:
            startLst.extend(self.split_akl((self.rak('N'), self.rak('CA'), self.rak('CB'))))
        if 'O' in self.akc:
            startLst.extend(self.split_akl((self.rak('O'), self.rak('C'), self.rak('CA'))))
        startLst.extend(NCaCKey)
        q = deque(startLst)
        if resetLocation:
            atomCoords = self._default_startpos()
        else:
            atomCoords = self._get_startpos()
        while q:
            '\n            if dbg:\n                print("assemble loop start q=", q)\n            '
            h1k = cast(HKT, q.pop())
            dihedraKeys = cic.id3_dh_index.get(h1k, None)
            '\n            if dbg:\n                print(\n                    "  h1k:",\n                    h1k,\n                    "len dihedra: ",\n                    len(dihedraKeys) if dihedraKeys is not None else "None",\n                )\n            '
            if dihedraKeys is not None:
                for dk in dihedraKeys:
                    d = cic.dihedra[dk]
                    dseqpos = int(d.atomkeys[0].akl[AtomKey.fields.respos])
                    d.initial_coords = cic.dAtoms[d.ndx]
                    if 4 == len(d.initial_coords) and d.initial_coords[3] is not None:
                        d_h2key = d.hedron2.atomkeys
                        ak = d.atomkeys[3]
                        '\n                        if dbg:\n                            print("    process", d, d_h2key, d.atomkeys)\n                        '
                        acount = len([a for a in d.atomkeys if a in atomCoords])
                        if 4 == acount:
                            if dseqpos == rseqpos:
                                q.appendleft(d_h2key)
                            '\n                            if dbg:\n                                print("    4- already done, append left")\n                            '
                            if not dcsValid[d.ndx]:
                                acs = [atomCoords[a] for a in h1k]
                                d.cst, d.rcst = coord_space(acs[0], acs[1], acs[2], True)
                                dcsValid[d.ndx] = True
                        elif 3 == acount:
                            '\n                            if dbg:\n                                print("    3- call coord_space")\n                            '
                            acs = np.asarray([atomCoords[a] for a in h1k])
                            d.cst, d.rcst = coord_space(acs[0], acs[1], acs[2], True)
                            dcsValid[d.ndx] = True
                            '\n                            if dbg:\n                                print("     acs:", acs.transpose())\n                                print("cst", d.cst)\n                                print("rcst", d.rcst)\n                                print(\n                                    "        initial_coords[3]=",\n                                    d.initial_coords[3].transpose(),\n                                )\n                            '
                            acak3 = d.rcst.dot(d.initial_coords[3])
                            '\n                            if dbg:\n                                print("        acak3=", acak3.transpose())\n                            '
                            atomCoords[ak] = acak3
                            aa[aaNdx[ak]] = acak3
                            aaValid[aaNdx[ak]] = True
                            '\n                            if dbg:\n                                print(\n                                    "        3- finished, ak:",\n                                    ak,\n                                    "coords:",\n                                    atomCoords[ak].transpose(),\n                                )\n                            '
                            if dseqpos == rseqpos:
                                q.appendleft(d_h2key)
                        elif verbose:
                            print('no coords to start', d)
                            print([a for a in d.atomkeys if atomCoords.get(a, None) is not None])
                    elif verbose:
                        print('no initial coords for', d)
        return atomCoords

    def split_akl(self, lst: Union[Tuple['AtomKey', ...], List['AtomKey']], missingOK: bool=False) -> List[Tuple['AtomKey', ...]]:
        """Get AtomKeys for this residue (ak_set) for generic list of AtomKeys.

        Changes and/or expands a list of 'generic' AtomKeys (e.g. 'N, C, C') to
        be specific to this Residue's altlocs etc., e.g.
        '(N-Ca_A_0.3-C, N-Ca_B_0.7-C)'

        Given a list of AtomKeys for a Hedron or Dihedron,
          return:
                list of matching atomkeys that have id3_dh in this residue
                (ak may change if occupancy != 1.00)

            or
                multiple lists of matching atomkeys expanded for all atom altlocs

            or
                empty list if any of atom_coord(ak) missing and not missingOK

        :param list lst: list[3] or [4] of AtomKeys.
            Non-altloc AtomKeys to match to specific AtomKeys for this residue
        :param bool missingOK: default False, see above.
        """
        altloc_ndx = AtomKey.fields.altloc
        occ_ndx = AtomKey.fields.occ
        edraLst: List[Tuple[AtomKey, ...]] = []
        altlocs = set()
        posnAltlocs: Dict['AtomKey', Set[str]] = {}
        akMap = {}
        for ak in lst:
            posnAltlocs[ak] = set()
            if ak in self.ak_set and ak.akl[altloc_ndx] is None and (ak.akl[occ_ndx] is None):
                edraLst.append((ak,))
            else:
                ak2_lst = []
                for ak2 in self.ak_set:
                    if ak.altloc_match(ak2):
                        ak2_lst.append(ak2)
                        akMap[ak2] = ak
                        altloc = ak2.akl[altloc_ndx]
                        if altloc is not None:
                            altlocs.add(altloc)
                            posnAltlocs[ak].add(altloc)
                edraLst.append(tuple(ak2_lst))
        maxc = 0
        for akl in edraLst:
            lenAKL = len(akl)
            if 0 == lenAKL and (not missingOK):
                return []
            elif maxc < lenAKL:
                maxc = lenAKL
        if 1 == maxc:
            newAKL = []
            for akl in edraLst:
                if akl:
                    newAKL.append(akl[0])
            return [tuple(newAKL)]
        else:
            new_edraLst = []
            for al in altlocs:
                alhl = []
                for akl in edraLst:
                    lenAKL = len(akl)
                    if 0 == lenAKL:
                        continue
                    if 1 == lenAKL:
                        alhl.append(akl[0])
                    elif al not in posnAltlocs[akMap[akl[0]]]:
                        alhl.append(sorted(akl)[0])
                    else:
                        for ak in akl:
                            if ak.akl[altloc_ndx] == al:
                                alhl.append(ak)
                new_edraLst.append(tuple(alhl))
            return new_edraLst

    def _gen_edra(self, lst: Union[Tuple['AtomKey', ...], List['AtomKey']]) -> None:
        """Populate hedra/dihedra given edron ID tuple.

        Given list of AtomKeys defining hedron or dihedron
          convert to AtomKeys with coordinates in this residue
          add appropriately to self.di/hedra, expand as needed atom altlocs

        :param list lst: tuple of AtomKeys.
            Specifies Hedron or Dihedron
        """
        for ak in lst:
            if ak.missing:
                return
        lenLst = len(lst)
        if 4 > lenLst:
            cdct, dct, obj = (self.cic.hedra, self.hedra, Hedron)
        else:
            cdct, dct, obj = (self.cic.dihedra, self.dihedra, Dihedron)
        if isinstance(lst, List):
            tlst = tuple(lst)
        else:
            tlst = lst
        hl = self.split_akl(tlst)
        for tnlst in hl:
            if len(tnlst) == lenLst:
                if tnlst not in cdct:
                    cdct[tnlst] = obj(tnlst)
                if tnlst not in dct:
                    dct[tnlst] = cdct[tnlst]
                dct[tnlst].needs_update = True

    def _create_edra(self, verbose: bool=False) -> None:
        """Create IC_Chain and IC_Residue di/hedra for atom coordinates.

        AllBonds handled here.

        :param bool verbose: default False.
            Warn about missing N, Ca, C backbone atoms.
        """
        if not self.ak_set:
            return
        sN, sCA, sC = (self.rak('N'), self.rak('CA'), self.rak('C'))
        if self.lc != 'G':
            sCB = self.rak('CB')
        if 0 < len(self.rnext) and self.rnext[0].ak_set:
            for rn in self.rnext:
                nN, nCA, nC = (rn.rak('N'), rn.rak('CA'), rn.rak('C'))
                nextNCaC = rn.split_akl((nN, nCA, nC), missingOK=True)
                for tpl in nextNCaC:
                    for ak in tpl:
                        if ak in rn.ak_set:
                            self.ak_set.add(ak)
                        else:
                            for rn_ak in rn.ak_set:
                                if rn_ak.altloc_match(ak):
                                    self.ak_set.add(rn_ak)
                self._gen_edra((sN, sCA, sC, nN))
                self._gen_edra((sCA, sC, nN, nCA))
                self._gen_edra((sC, nN, nCA, nC))
                self._gen_edra((sCA, sC, nN))
                self._gen_edra((sC, nN, nCA))
                self._gen_edra((nN, nCA, nC))
                try:
                    nO = rn.akc['O']
                except KeyError:
                    nCB = rn.akc.get('CB', None)
                    if nCB is not None and nCB in rn.ak_set:
                        self.ak_set.add(nCB)
                        self._gen_edra((nN, nCA, nCB))
                        self._gen_edra((sC, nN, nCA, nCB))
        if 0 == len(self.rprev):
            self._gen_edra((sN, sCA, sC))
        backbone = ic_data_backbone
        for edra in backbone:
            if all((atm in self.akc for atm in edra)):
                r_edra = [self.rak(atom) for atom in edra]
                self._gen_edra(r_edra)
        if self.lc is not None:
            sidechain = ic_data_sidechains.get(self.lc, [])
            for edraLong in sidechain:
                edra = edraLong[0:4]
                if all((atm in self.akc for atm in edra)):
                    r_edra = [self.rak(atom) for atom in edra]
                    self._gen_edra(r_edra)
            if IC_Residue._AllBonds:
                sidechain = ic_data_sidechain_extras.get(self.lc, [])
                for edra in sidechain:
                    if all((atm in self.akc for atm in edra)):
                        r_edra = [self.rak(atom) for atom in edra]
                        self._gen_edra(r_edra)
        if self.gly_Cbeta and 'G' == self.lc:
            self.ak_set.add(AtomKey(self, 'CB'))
            sCB = self.rak('CB')
            sCB.missing = False
            self.cic.akset.add(sCB)
            sO = self.rak('O')
            htpl = (sCB, sCA, sC)
            self._gen_edra(htpl)
            dtpl = (sO, sC, sCA, sCB)
            self._gen_edra(dtpl)
            d = self.dihedra[dtpl]
            d.ric = self
            d._set_hedra()
            if not hasattr(self.cic, 'gcb'):
                self.cic.gcb = {}
            self.cic.gcb[sCB] = dtpl
        self._link_dihedra(verbose)
        if verbose:
            self.rak('O')
            missing = []
            for akk, akv in self.akc.items():
                if isinstance(akk, str) and akv.missing:
                    missing.append(akv)
            if missing:
                chn = self.residue.parent
                chn_id = chn.id
                chn_len = len(chn.internal_coord.ordered_aa_ic_list)
                print(f'chain {chn_id} len {chn_len} missing atom(s): {missing}')
    atom_sernum = None
    atom_chain = None

    @staticmethod
    def _pdb_atom_string(atm: Atom, cif_extend: bool=False) -> str:
        """Generate PDB ATOM record.

        :param Atom atm: Biopython Atom object reference
        :param IC_Residue.atom_sernum: Class variable default None.
            override atom serial number if not None
        :param IC_Residue.atom_chain: Class variable default None.
            override atom chain id if not None
        """
        if 2 == atm.is_disordered():
            if IC_Residue.no_altloc:
                return IC_Residue._pdb_atom_string(atm.selected_child, cif_extend)
            s = ''
            for a in atm.child_dict.values():
                s += IC_Residue._pdb_atom_string(a, cif_extend)
            return s
        else:
            res = atm.parent
            chn = res.parent
            fmt = '{:6}{:5d} {:4}{:1}{:3} {:1}{:4}{:1}   {:8.3f}{:8.3f}{:8.3f}{:6.2f}{:6.2f}        {:>4}\n'
            if cif_extend:
                fmt = '{:6}{:5d} {:4}{:1}{:3} {:1}{:4}{:1}   {:10.5f}{:10.5f}{:10.5f}{:7.3f}{:6.2f}        {:>4}\n'
            s = fmt.format('ATOM', IC_Residue.atom_sernum if IC_Residue.atom_sernum is not None else atm.serial_number, atm.fullname, atm.altloc, res.resname, IC_Residue.atom_chain if IC_Residue.atom_chain is not None else chn.id, res.id[1], res.id[2], atm.coord[0], atm.coord[1], atm.coord[2], atm.occupancy, atm.bfactor, atm.element)
        return s

    def pdb_residue_string(self) -> str:
        """Generate PDB ATOM records for this residue as string.

        Convenience method for functionality not exposed in PDBIO.py.
        Increments :data:`IC_Residue.atom_sernum` if not None

        :param IC_Residue.atom_sernum: Class variable default None.
            Override and increment atom serial number if not None
        :param IC_Residue.atom_chain: Class variable.
            Override atom chain id if not None

        .. todo::
            move to PDBIO
        """
        str = ''
        atomArrayIndex = self.cic.atomArrayIndex
        bpAtomArray = self.cic.bpAtomArray
        respos = self.rbase[0]
        resposNdx = AtomKey.fields.respos
        for ak in sorted(self.ak_set):
            if int(ak.akl[resposNdx]) == respos:
                str += IC_Residue._pdb_atom_string(bpAtomArray[atomArrayIndex[ak]])
                if IC_Residue.atom_sernum is not None:
                    IC_Residue.atom_sernum += 1
        return str

    @staticmethod
    def _residue_string(res: 'Residue') -> str:
        """Generate PIC Residue string.

        Enough to create Biopython Residue object without actual Atoms.

        :param Residue res: Biopython Residue object reference
        """
        segid = res.get_segid()
        if segid.isspace() or '' == segid:
            segid = ''
        else:
            segid = ' [' + segid + ']'
        return str(res.get_full_id()) + ' ' + res.resname + segid + '\n'
    _pfDef = namedtuple('_pfDef', ['psi', 'omg', 'phi', 'tau', 'chi1', 'chi2', 'chi3', 'chi4', 'chi5', 'pomg', 'chi', 'classic_b', 'classic', 'hedra', 'primary', 'secondary', 'all', 'initAtoms', 'bFactors'])
    _b = [1 << i for i in range(16)]
    _bChi = _b[4] | _b[5] | _b[6] | _b[7] | _b[8]
    _bClassB = _b[0] | _b[2] | _b[3] | _b[9]
    _bClass = _bClassB | _bChi
    _bAll = _b[10] | _b[11] | _b[12]
    pic_flags = _pfDef(_b[0], _b[1], _b[2], _b[3], _b[4], _b[5], _b[6], _b[7], _b[8], _b[9], _bChi, _bClassB, _bClass, _b[10], _b[11], _b[12], _bAll, _b[13], _b[14])
    'Used by :func:`.PICIO.write_PIC` to control classes of values to be defaulted.'
    picFlagsDefault = pic_flags.all | pic_flags.initAtoms | pic_flags.bFactors
    'Default is all dihedra + initial tau atoms + bFactors.'
    picFlagsDict = pic_flags._asdict()
    'Dictionary of pic_flags values to use as needed.'

    def _write_pic_bfac(self, atm: Atom, s: str, col: int) -> Tuple[str, int]:
        ak = self.rak(atm)
        if 0 == col % 5:
            s += 'BFAC:'
        s += ' ' + ak.id + ' ' + f'{atm.get_bfactor():6.2f}'
        col += 1
        if 0 == col % 5:
            s += '\n'
        return (s, col)

    def _write_PIC(self, pdbid: str='0PDB', chainid: str='A', picFlags: int=picFlagsDefault, hCut: Optional[Union[float, None]]=None, pCut: Optional[Union[float, None]]=None) -> str:
        """Write PIC format lines for this residue.

        See :func:`.PICIO.write_PIC`.

        :param str pdbid: PDB idcode string; default 0PDB
        :param str chainid: PDB Chain ID character; default A
        :param int picFlags: control details written to PIC file; see
            :meth:`.PICIO.write_PIC`
        :param float hCut: only write hedra with ref db angle std dev > this
            value; default None
        :param float pCut: only write primary dihedra with ref db angle
            std dev > this value; default None
        """
        pAcc = IC_Residue.pic_accuracy
        if pdbid is None:
            pdbid = '0PDB'
        if chainid is None:
            chainid = 'A'
        icr = IC_Residue
        s = icr._residue_string(self.residue)
        if picFlags & icr.pic_flags.initAtoms and 0 == len(self.rprev) and hasattr(self, 'NCaCKey') and (self.NCaCKey is not None) and (not np.all(self.residue['N'].coord == self.residue['N'].coord[0])):
            NCaChedron = self.pick_angle(self.NCaCKey[0])
            if NCaChedron is not None:
                try:
                    ts = IC_Residue._pdb_atom_string(self.residue['N'], cif_extend=True)
                    ts += IC_Residue._pdb_atom_string(self.residue['CA'], cif_extend=True)
                    ts += IC_Residue._pdb_atom_string(self.residue['C'], cif_extend=True)
                    s += ts
                except KeyError:
                    pass
        base = pdbid + ' ' + chainid + ' '
        cic = self.cic
        if picFlags & icr.pic_flags.hedra or picFlags & icr.pic_flags.tau:
            for h in sorted(self.hedra.values()):
                if not picFlags & icr.pic_flags.hedra and picFlags & icr.pic_flags.tau and (h.e_class != 'NCAC'):
                    continue
                if hCut is not None:
                    hc = h.xrh_class if hasattr(h, 'xrh_class') else h.e_class
                    if hc in hedra_defaults and hedra_defaults[hc][1] <= hCut:
                        continue
                hndx = h.ndx
                try:
                    s += base + h.id + ' ' + f'{cic.hedraL12[hndx]:{pAcc}} {cic.hedraAngle[hndx]:{pAcc}} {cic.hedraL23[hndx]:{pAcc}}' + '\n'
                except KeyError:
                    pass
        for d in sorted(self.dihedra.values()):
            if d.primary:
                if not picFlags & icr.pic_flags.primary:
                    if not picFlags & d.bits():
                        continue
            elif not picFlags & icr.pic_flags.secondary:
                continue
            if pCut is not None:
                if d.primary and d.pclass in dihedra_primary_defaults and (dihedra_primary_defaults[d.pclass][1] <= pCut):
                    continue
            try:
                s += base + d.id + ' ' + f'{cic.dihedraAngle[d.ndx]:{pAcc}}' + '\n'
            except KeyError:
                pass
        if picFlags & icr.pic_flags.bFactors:
            col = 0
            for a in sorted(self.residue.get_atoms()):
                if 2 == a.is_disordered():
                    if IC_Residue.no_altloc or self.alt_ids is None:
                        s, col = self._write_pic_bfac(a.selected_child, s, col)
                    else:
                        for atm in a.child_dict.values():
                            s, col = self._write_pic_bfac(atm, s, col)
                else:
                    s, col = self._write_pic_bfac(a, s, col)
            if 0 != col % 5:
                s += '\n'
        return s

    def _get_ak_tuple(self, ak_str: str) -> Optional[Tuple['AtomKey', ...]]:
        """Convert atom pair string to AtomKey tuple.

        :param str ak_str:
            Two atom names separated by ':', e.g. 'N:CA'
            Optional position specifier relative to self,
            e.g. '-1C:N' for preceding peptide bond.
        """
        AK = AtomKey
        S = self
        angle_key2 = []
        akstr_list = ak_str.split(':')
        lenInput = len(akstr_list)
        for a in akstr_list:
            m = self._relative_atom_re.match(a)
            if m:
                if m.group(1) == '-1':
                    if 0 < len(S.rprev):
                        angle_key2.append(AK(S.rprev[0], m.group(2)))
                elif m.group(1) == '1':
                    if 0 < len(S.rnext):
                        angle_key2.append(AK(S.rnext[0], m.group(2)))
                elif m.group(1) == '0':
                    angle_key2.append(self.rak(m.group(2)))
            else:
                angle_key2.append(self.rak(a))
        if len(angle_key2) != lenInput:
            return None
        return tuple(angle_key2)
    _relative_atom_re = re.compile('^(-?[10])([A-Z]+)$')

    def _get_angle_for_tuple(self, angle_key: EKT) -> Optional[Union['Hedron', 'Dihedron']]:
        len_mkey = len(angle_key)
        rval: Optional[Union['Hedron', 'Dihedron']]
        if 4 == len_mkey:
            rval = self.dihedra.get(cast(DKT, angle_key), None)
        elif 3 == len_mkey:
            rval = self.hedra.get(cast(HKT, angle_key), None)
        else:
            return None
        return rval

    def pick_angle(self, angle_key: Union[EKT, str]) -> Optional[Union['Hedron', 'Dihedron']]:
        """Get Hedron or Dihedron for angle_key.

        :param angle_key:
            - tuple of 3 or 4 AtomKeys
            - string of atom names ('CA') separated by :'s
            - string of [-1, 0, 1]<atom name> separated by ':'s. -1 is
              previous residue, 0 is this residue, 1 is next residue
            - psi, phi, omg, omega, chi1, chi2, chi3, chi4, chi5
            - tau (N-CA-C angle) see Richardson1981
            - tuples of AtomKeys is only access for alternate disordered atoms

        Observe that a residue's phi and omega dihedrals, as well as the hedra
        comprising them (including the N:Ca:C `tau` hedron), are stored in the
        n-1 di/hedra sets; this overlap is handled here, but may be an issue if
        accessing directly.

        The following print commands are equivalent (except for sidechains with
        non-carbon atoms for chi2)::

            ric = r.internal_coord
            print(
                r,
                ric.get_angle("psi"),
                ric.get_angle("phi"),
                ric.get_angle("omg"),
                ric.get_angle("tau"),
                ric.get_angle("chi2"),
            )
            print(
                r,
                ric.get_angle("N:CA:C:1N"),
                ric.get_angle("-1C:N:CA:C"),
                ric.get_angle("-1CA:-1C:N:CA"),
                ric.get_angle("N:CA:C"),
                ric.get_angle("CA:CB:CG:CD"),
            )

        See ic_data.py for detail of atoms in the enumerated sidechain angles
        and the backbone angles which do not span the peptide bond. Using 's'
        for current residue ('self') and 'n' for next residue, the spanning
        (overlapping) angles are::

                (sN, sCA, sC, nN)   # psi
                (sCA, sC, nN, nCA)  # omega i+1
                (sC, nN, nCA, nC)   # phi i+1
                (sCA, sC, nN)
                (sC, nN, nCA)
                (nN, nCA, nC)       # tau i+1

        :return: Matching Hedron, Dihedron, or None.
        """
        rval: Optional[Union['Hedron', 'Dihedron']] = None
        if isinstance(angle_key, tuple):
            rval = self._get_angle_for_tuple(angle_key)
            if rval is None and self.rprev:
                rval = self.rprev[0]._get_angle_for_tuple(angle_key)
        elif ':' in angle_key:
            angle_key = cast(EKT, self._get_ak_tuple(cast(str, angle_key)))
            if angle_key is None:
                return None
            rval = self._get_angle_for_tuple(angle_key)
            if rval is None and self.rprev:
                rval = self.rprev[0]._get_angle_for_tuple(angle_key)
        elif 'psi' == angle_key:
            if 0 == len(self.rnext):
                return None
            rn = self.rnext[0]
            sN, sCA, sC = (self.rak('N'), self.rak('CA'), self.rak('C'))
            nN = rn.rak('N')
            rval = self.dihedra.get((sN, sCA, sC, nN), None)
        elif 'phi' == angle_key:
            if 0 == len(self.rprev):
                return None
            rp = self.rprev[0]
            pC, sN, sCA = (rp.rak('C'), self.rak('N'), self.rak('CA'))
            sC = self.rak('C')
            rval = rp.dihedra.get((pC, sN, sCA, sC), None)
        elif 'omg' == angle_key or 'omega' == angle_key:
            if 0 == len(self.rprev):
                return None
            rp = self.rprev[0]
            pCA, pC, sN = (rp.rak('CA'), rp.rak('C'), self.rak('N'))
            sCA = self.rak('CA')
            rval = rp.dihedra.get((pCA, pC, sN, sCA), None)
        elif 'tau' == angle_key:
            sN, sCA, sC = (self.rak('N'), self.rak('CA'), self.rak('C'))
            rval = self.hedra.get((sN, sCA, sC), None)
            if rval is None and 0 != len(self.rprev):
                rp = self.rprev[0]
                rval = rp.hedra.get((sN, sCA, sC), None)
        elif angle_key.startswith('chi'):
            sclist = ic_data_sidechains.get(self.lc, None)
            if sclist is None:
                return None
            ndx = 2 * int(angle_key[-1]) - 1
            try:
                akl = sclist[ndx]
                if akl[4] == angle_key:
                    klst = [self.rak(a) for a in akl[0:4]]
                    tklst = cast(DKT, tuple(klst))
                    rval = self.dihedra.get(tklst, None)
                else:
                    return None
            except IndexError:
                return None
        return rval

    def get_angle(self, angle_key: Union[EKT, str]) -> Optional[float]:
        """Get dihedron or hedron angle for specified key.

        See :meth:`.pick_angle` for key specifications.
        """
        edron = self.pick_angle(angle_key)
        if edron:
            return edron.angle
        return None

    def set_angle(self, angle_key: Union[EKT, str], v: float, overlap=True):
        """Set dihedron or hedron angle for specified key.

        If angle is a `Dihedron` and `overlap` is True (default), overlapping
        dihedra are also changed as appropriate.  The overlap is a result of
        protein chain definitions in :mod:`.ic_data` and :meth:`_create_edra`
        (e.g. psi overlaps N-CA-C-O).

        Te default overlap=True is probably what you want for:
        `set_angle("chi1", val)`

        The default is probably NOT what you want when processing all dihedrals
        in a chain or residue (such as copying from another structure), as the
        overlaping dihedra will likely be in the set as well.

        N.B. setting e.g. PRO chi2 is permitted without error or warning!

        See :meth:`.pick_angle` for angle_key specifications.
        See :meth:`.bond_rotate` to change a dihedral by a number of degrees

        :param angle_key: angle identifier.
        :param float v: new angle in degrees (result adjusted to +/-180).
        :param bool overlap: default True.
            Modify overlapping dihedra as needed
        """
        edron = self.pick_angle(angle_key)
        if edron is None:
            return
        elif isinstance(edron, Hedron) or not overlap:
            edron.angle = v
        else:
            delta = Dihedron.angle_dif(edron.angle, v)
            self._do_bond_rotate(edron, delta)

    def _do_bond_rotate(self, base: 'Dihedron', delta: float):
        """Find and modify related dihedra through id3_dh_index."""
        try:
            for dk in self.cic.id3_dh_index[base.id3]:
                dihed = self.cic.dihedra[dk]
                dihed.angle += delta
                try:
                    for d2rk in self.cic.id3_dh_index[dihed.id32[::-1]]:
                        self.cic.dihedra[d2rk].angle += delta
                except KeyError:
                    pass
        except AttributeError:
            raise RuntimeError('bond_rotate, bond_set only for dihedral angles')

    def bond_rotate(self, angle_key: Union[EKT, str], delta: float):
        """Rotate set of overlapping dihedrals by delta degrees.

        Changes a dihedral angle by a given delta, i.e.
        new_angle = current_angle + delta
        Values are adjusted so new_angle iwll be within +/-180.

        Changes overlapping dihedra as in :meth:`.set_angle`

        See :meth:`.pick_angle` for key specifications.
        """
        base = self.pick_angle(angle_key)
        if base is not None:
            self._do_bond_rotate(base, delta)

    def bond_set(self, angle_key: Union[EKT, str], val: float):
        """Set dihedron to val, update overlapping dihedra by same amount.

        Redundant to :meth:`.set_angle`, retained for compatibility.  Unlike
        :meth:`.set_angle` this is for dihedra only and no option to not update
        overlapping dihedra.

        See :meth:`.pick_angle` for key specifications.
        """
        base = self.pick_angle(angle_key)
        if base is not None:
            delta = Dihedron.angle_dif(base.angle, val)
            self._do_bond_rotate(base, delta)

    def pick_length(self, ak_spec: Union[str, BKT]) -> Tuple[Optional[List['Hedron']], Optional[BKT]]:
        """Get list of hedra containing specified atom pair.

        :param ak_spec:
            - tuple of two AtomKeys
            - string: two atom names separated by ':', e.g. 'N:CA' with
              optional position specifier relative to self, e.g. '-1C:N' for
              preceding peptide bond.  Position specifiers are -1, 0, 1.

        The following are equivalent::

            ric = r.internal_coord
            print(
                r,
                ric.get_length("0C:1N"),
            )
            print(
                r,
                None
                if not ric.rnext
                else ric.get_length((ric.rak("C"), ric.rnext[0].rak("N"))),
            )

        If atom not found on current residue then will look on rprev[0] to
        handle cases like Gly N:CA.  For finer control please access
        `IC_Chain.hedra` directly.

        :return: list of hedra containing specified atom pair as tuples of
                AtomKeys
        """
        rlst: List[Hedron] = []
        if isinstance(ak_spec, str):
            ak_spec = cast(BKT, self._get_ak_tuple(ak_spec))
        if ak_spec is None:
            return (None, None)
        for hed_key, hed_val in self.hedra.items():
            if all((ak in hed_key for ak in ak_spec)):
                rlst.append(hed_val)
        for rp in self.rprev:
            for hed_key, hed_val in rp.hedra.items():
                if all((ak in hed_key for ak in ak_spec)):
                    rlst.append(hed_val)
        return (rlst, ak_spec)

    def get_length(self, ak_spec: Union[str, BKT]) -> Optional[float]:
        """Get bond length for specified atom pair.

        See :meth:`.pick_length` for ak_spec and details.
        """
        hed_lst, ak_spec2 = self.pick_length(ak_spec)
        if hed_lst is None or ak_spec2 is None:
            return None
        for hed in hed_lst:
            val = hed.get_length(ak_spec2)
            if val is not None:
                return val
        return None

    def set_length(self, ak_spec: Union[str, BKT], val: float) -> None:
        """Set bond length for specified atom pair.

        See :meth:`.pick_length` for ak_spec.
        """
        hed_lst, ak_spec2 = self.pick_length(ak_spec)
        if hed_lst is not None and ak_spec2 is not None:
            for hed in hed_lst:
                hed.set_length(ak_spec2, val)

    def applyMtx(self, mtx: np.array) -> None:
        """Apply matrix to atom_coords for this IC_Residue."""
        aa = self.cic.atomArray
        aai = self.cic.atomArrayIndex
        rpndx = AtomKey.fields.respos
        rp = str(self.rbase[0])
        aselect = [aai.get(k) for k in aai.keys() if k.akl[rpndx] == rp]
        aas = aa[aselect]
        aa[aselect] = aas.dot(mtx.transpose())
        '\n        # slower way, one at a time\n        for ak in sorted(self.ak_set):\n            ndx = self.cic.atomArrayIndex[ak]\n            self.cic.atomArray[ndx] = mtx.dot(self.cic.atomArray[ndx])\n        '