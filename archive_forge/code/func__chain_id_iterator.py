import itertools
from collections import defaultdict
from string import ascii_uppercase
from Bio.PDB.StructureBuilder import StructureBuilder
from Bio.PDB.PDBIO import Select, StructureIO
from mmtf.api.mmtf_writer import MMTFEncoder
from Bio.SeqUtils import seq1
from Bio.Data.PDBData import protein_letters_3to1_extended
def _chain_id_iterator(self):
    """Label chains sequentially: A, B, ..., Z, AA, AB etc."""
    for size in itertools.count(1):
        for s in itertools.product(ascii_uppercase, repeat=size):
            yield ''.join(s)