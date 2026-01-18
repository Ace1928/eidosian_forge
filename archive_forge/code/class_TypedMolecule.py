import sys
import copy
import itertools
import re
import time
import weakref
from collections import Counter, defaultdict, namedtuple
from heapq import heapify, heappop, heappush
from itertools import chain, combinations
class TypedMolecule(object):

    def __init__(self, rdmol, rdmol_atoms, rdmol_bonds, atom_smarts_types, bond_smarts_types, canonical_bondtypes):
        self.rdmol = rdmol
        self.rdmol_atoms = rdmol_atoms
        self.rdmol_bonds = rdmol_bonds
        self.atom_smarts_types = atom_smarts_types
        self.bond_smarts_types = bond_smarts_types
        self.canonical_bondtypes = canonical_bondtypes