import collections
import warnings
from Bio import BiopythonParserWarning
from Bio.Data.PDBData import protein_letters_3to1
from Bio.Data.PDBData import protein_letters_3to1_extended
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from .Interfaces import SequenceIterator
def PdbAtomIterator(source):
    """Return SeqRecord objects for each chain in a PDB file.

    Argument source is a file-like object or a path to a file.

    The sequences are derived from the 3D structure (ATOM records), not the
    SEQRES lines in the PDB file header.

    Unrecognised three letter amino acid codes (e.g. "CSD") from HETATM entries
    are converted to "X" in the sequence.

    In addition to information from the PDB header (which is the same for all
    records), the following chain specific information is placed in the
    annotation:

    record.annotations["residues"] = List of residue ID strings
    record.annotations["chain"] = Chain ID (typically A, B ,...)
    record.annotations["model"] = Model ID (typically zero)

    Where amino acids are missing from the structure, as indicated by residue
    numbering, the sequence is filled in with 'X' characters to match the size
    of the missing region, and  None is included as the corresponding entry in
    the list record.annotations["residues"].

    This function uses the Bio.PDB module to do most of the hard work. The
    annotation information could be improved but this extra parsing should be
    done in parse_pdb_header, not this module.

    This gets called internally via Bio.SeqIO for the atom based interpretation
    of the PDB file format:

    >>> from Bio import SeqIO
    >>> for record in SeqIO.parse("PDB/1A8O.pdb", "pdb-atom"):
    ...     print("Record id %s, chain %s" % (record.id, record.annotations["chain"]))
    ...
    Record id 1A8O:A, chain A

    Equivalently,

    >>> with open("PDB/1A8O.pdb") as handle:
    ...     for record in PdbAtomIterator(handle):
    ...         print("Record id %s, chain %s" % (record.id, record.annotations["chain"]))
    ...
    Record id 1A8O:A, chain A

    """
    from Bio.PDB import PDBParser
    structure = PDBParser().get_structure(None, source)
    pdb_id = structure.header['idcode']
    if not pdb_id:
        warnings.warn("'HEADER' line not found; can't determine PDB ID.", BiopythonParserWarning)
        pdb_id = '????'
    for record in AtomIterator(pdb_id, structure):
        record.annotations.update(structure.header)
        yield record