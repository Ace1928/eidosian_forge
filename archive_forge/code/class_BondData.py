import time
import numpy as np
from ase.atom import Atom
from ase.atoms import Atoms
from ase.calculators.lammpsrun import Prism
from ase.neighborlist import NeighborList
from ase.data import atomic_masses, chemical_symbols
from ase.io import read
class BondData:

    def __init__(self, name_value_hash):
        self.nvh = name_value_hash

    def name_value(self, aname, bname):
        name1 = twochar(aname) + '-' + twochar(bname)
        name2 = twochar(bname) + '-' + twochar(aname)
        if name1 in self.nvh:
            return (name1, self.nvh[name1])
        if name2 in self.nvh:
            return (name2, self.nvh[name2])
        return (None, None)

    def value(self, aname, bname):
        return self.name_value(aname, bname)[1]