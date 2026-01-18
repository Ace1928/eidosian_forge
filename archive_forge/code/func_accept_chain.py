import re
import warnings
from Bio.PDB.PDBIO import PDBIO
from Bio import BiopythonWarning
def accept_chain(self, chain):
    """Verify if chain match chain identifier."""
    if chain.get_id() == self.chain_id:
        return 1
    return 0