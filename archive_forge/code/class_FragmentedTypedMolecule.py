import sys
import copy
import itertools
import re
import time
import weakref
from collections import Counter, defaultdict, namedtuple
from heapq import heapify, heappop, heappush
from itertools import chain, combinations
class FragmentedTypedMolecule(object):

    def __init__(self, rdmol, rdmol_atoms, orig_atoms, orig_bonds, atom_smarts_types, bond_smarts_types, canonical_bondtypes):
        self.rdmol = rdmol
        self.rdmol_atoms = rdmol_atoms
        self.orig_atoms = orig_atoms
        self.orig_bonds = orig_bonds
        self.atom_smarts_types = atom_smarts_types
        self.bond_smarts_types = bond_smarts_types
        self.canonical_bondtypes = canonical_bondtypes