import re
import warnings
from Bio.PDB.PDBIO import PDBIO
from Bio import BiopythonWarning
def accept_model(self, model):
    """Verify if model match the model identifier."""
    if model.get_id() == self.model_id:
        return 1
    return 0