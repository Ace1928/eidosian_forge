import re
from collections import defaultdict
from Bio.PDB.StructureBuilder import StructureBuilder
from Bio.PDB.PDBIO import Select, StructureIO
def _get_label_asym_id(self, entity_id):
    div = entity_id
    out = ''
    while div > 0:
        mod = (div - 1) % 26
        out += chr(65 + mod)
        div = int((div - mod) / 26)
    return out