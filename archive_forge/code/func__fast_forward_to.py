import numpy as np
from itertools import islice
import os
from ase.atoms import Atoms, Atom
from ase.cell import Cell
from ase.io.formats import index2range
from ase.data import atomic_numbers
def _fast_forward_to(fileobj, section_header):
    """Helper to forward to a section"""
    found = False
    while fileobj:
        line = fileobj.readline()
        if section_header in line:
            found = True
            break
    if not found:
        raise RuntimeError('No {:} section found!'.format(section_header))