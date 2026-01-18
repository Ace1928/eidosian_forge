import collections
import warnings
from Bio import BiopythonParserWarning
from Bio.Data.PDBData import protein_letters_3to1
from Bio.Data.PDBData import protein_letters_3to1_extended
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from .Interfaces import SequenceIterator
def _res2aacode(residue, undef_code='X'):
    """Return the one-letter amino acid code from the residue name.

    Non-amino acid are returned as "X".
    """
    if isinstance(residue, str):
        return _aa3to1_dict.get(residue, undef_code)
    return _aa3to1_dict.get(residue.resname, undef_code)