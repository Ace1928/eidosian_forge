import numpy as np
from itertools import islice
import os
from ase.atoms import Atoms, Atom
from ase.cell import Cell
from ase.io.formats import index2range
from ase.data import atomic_numbers
def _parse_section(inp):
    """Helper to parse structure to nested dict"""
    ret = {'content': []}
    while inp:
        line = inp.readline().strip()
        if line.startswith('&END'):
            return ret
        elif line.startswith('&'):
            key = line.replace('&', '')
            ret[key] = _parse_section(inp)
        else:
            ret['content'].append(line)
    return ret