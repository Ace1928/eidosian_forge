import numpy as np
from itertools import islice
import os
from ase.atoms import Atoms, Atom
from ase.cell import Cell
from ase.io.formats import index2range
from ase.data import atomic_numbers
class DCDChunk:

    def __init__(self, chunk, dtype, natoms, symbols, aligned):
        self.chunk = chunk
        self.dtype = dtype
        self.natoms = natoms
        self.symbols = symbols
        self.aligned = aligned

    def build(self):
        """Convert unprocessed chunk into Atoms."""
        return _read_cp2k_dcd_frame(self.chunk, self.dtype, self.natoms, self.symbols, self.aligned)