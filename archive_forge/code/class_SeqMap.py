from copy import copy
from Bio.Data.PDBData import protein_letters_3to1_extended
from Bio.SCOP.Residues import Residues
class SeqMap:
    """An ASTRAL RAF (Rapid Access Format) Sequence Map.

    This is a list like object; You can find the location of particular residues
    with index(), slice this SeqMap into fragments, and glue fragments back
    together with extend().

    Attributes:
     - pdbid -- The PDB 4 character ID
     - pdb_datestamp -- From the PDB file
     - version -- The RAF format version. e.g. 0.01
     - flags -- RAF flags. (See release notes for more information.)
     - res -- A list of Res objects, one for each residue in this sequence map

    """

    def __init__(self, line=None):
        """Initialize the class."""
        self.pdbid = ''
        self.pdb_datestamp = ''
        self.version = ''
        self.flags = ''
        self.res = []
        if line:
            self._process(line)

    def _process(self, line):
        """Parse a RAF record into a SeqMap object (PRIVATE)."""
        header_len = 38
        line = line.rstrip()
        if len(line) < header_len:
            raise ValueError('Incomplete header: ' + line)
        self.pdbid = line[0:4]
        chainid = line[4:5]
        self.version = line[6:10]
        if self.version != '0.01' and self.version != '0.02':
            raise ValueError('Incompatible RAF version: ' + self.version)
        self.pdb_datestamp = line[14:20]
        self.flags = line[21:27]
        for i in range(header_len, len(line), 7):
            f = line[i:i + 7]
            if len(f) != 7:
                raise ValueError('Corrupt Field: (' + f + ')')
            r = Res()
            r.chainid = chainid
            r.resid = f[0:5].strip()
            r.atom = normalize_letters(f[5:6])
            r.seqres = normalize_letters(f[6:7])
            self.res.append(r)

    def index(self, resid, chainid='_'):
        """Return the index of the SeqMap for the given resid and chainid."""
        for i in range(len(self.res)):
            if self.res[i].resid == resid and self.res[i].chainid == chainid:
                return i
        raise KeyError('No such residue ' + chainid + resid)

    def __getitem__(self, index):
        """Extract a single Res object from the SeqMap."""
        if not isinstance(index, slice):
            raise NotImplementedError
        s = copy(self)
        s.res = s.res[index]
        return s

    def append(self, res):
        """Append another Res object onto the list of residue mappings."""
        self.res.append(res)

    def extend(self, other):
        """Append another SeqMap onto the end of self.

        Both SeqMaps must have the same PDB ID, PDB datestamp and
        RAF version.  The RAF flags are erased if they are inconsistent. This
        may happen when fragments are taken from different chains.
        """
        if not isinstance(other, SeqMap):
            raise TypeError('Can only extend a SeqMap with a SeqMap.')
        if self.pdbid != other.pdbid:
            raise TypeError('Cannot add fragments from different proteins')
        if self.version != other.version:
            raise TypeError('Incompatible rafs')
        if self.pdb_datestamp != other.pdb_datestamp:
            raise TypeError('Different pdb dates!')
        if self.flags != other.flags:
            self.flags = ''
        self.res += other.res

    def __iadd__(self, other):
        """In place addition of SeqMap objects."""
        self.extend(other)
        return self

    def __add__(self, other):
        """Addition of SeqMap objects."""
        s = copy(self)
        s.extend(other)
        return s

    def getAtoms(self, pdb_handle, out_handle):
        """Extract all relevant ATOM and HETATOM records from a PDB file.

        The PDB file is scanned for ATOM and HETATOM records. If the
        chain ID, residue ID (seqNum and iCode), and residue type match
        a residue in this sequence map, then the record is echoed to the
        output handle.

        This is typically used to find the coordinates of a domain, or other
        residue subset.

        Arguments:
         - pdb_handle -- A handle to the relevant PDB file.
         - out_handle -- All output is written to this file like object.

        """
        resSet = {}
        for r in self.res:
            if r.atom == 'X':
                continue
            chainid = r.chainid
            if chainid == '_':
                chainid = ' '
            resid = r.resid
            resSet[chainid, resid] = r
        resFound = {}
        for line in pdb_handle:
            if line.startswith(('ATOM  ', 'HETATM')):
                chainid = line[21:22]
                resid = line[22:27].strip()
                key = (chainid, resid)
                if key in resSet:
                    res = resSet[key]
                    atom_aa = res.atom
                    resName = line[17:20]
                    if resName in protein_letters_3to1_extended:
                        if protein_letters_3to1_extended[resName] == atom_aa:
                            out_handle.write(line)
                            resFound[key] = res
        if len(resSet) != len(resFound):
            raise RuntimeError('Could not find at least one ATOM or HETATM record for each and every residue in this sequence map.')