import re
from collections import defaultdict
from Bio.PDB.StructureBuilder import StructureBuilder
from Bio.PDB.PDBIO import Select, StructureIO
def _format_mmcif_col(self, val, col_width):
    if self._requires_newline(val):
        return '\n;' + val + '\n;\n'
    elif self._requires_quote(val):
        if "' " in val:
            return '{v: <{width}}'.format(v='"' + val + '"', width=col_width)
        else:
            return '{v: <{width}}'.format(v="'" + val + "'", width=col_width)
    else:
        return '{v: <{width}}'.format(v=val, width=col_width)