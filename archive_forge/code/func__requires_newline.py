import re
from collections import defaultdict
from Bio.PDB.StructureBuilder import StructureBuilder
from Bio.PDB.PDBIO import Select, StructureIO
def _requires_newline(self, val):
    if '\n' in val or ("' " in val and '" ' in val):
        return True
    else:
        return False