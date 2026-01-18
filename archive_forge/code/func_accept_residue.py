import re
import warnings
from Bio.PDB.PDBIO import PDBIO
from Bio import BiopythonWarning
def accept_residue(self, residue):
    """Verify if a residue sequence is between the start and end sequence."""
    hetatm_flag, resseq, icode = residue.get_id()
    if hetatm_flag != ' ':
        return 0
    if icode != ' ':
        warnings.warn(f'WARNING: Icode {icode} at position {resseq}', BiopythonWarning)
    if self.start <= resseq <= self.end:
        return 1
    return 0