import os
import subprocess
import tempfile
import warnings
import numpy as np
from Bio.PDB import PDBParser
from Bio.PDB import Selection
from Bio.PDB.AbstractPropertyMap import AbstractPropertyMap
from Bio.PDB.Polypeptide import is_aa
from Bio import BiopythonWarning
def _read_vertex_array(filename):
    """Read the vertex list into a NumPy array (PRIVATE)."""
    with open(filename) as fp:
        vertex_list = []
        for line in fp:
            sl = line.split()
            if len(sl) != 9:
                continue
            vl = [float(x) for x in sl[0:3]]
            vertex_list.append(vl)
    return np.array(vertex_list)