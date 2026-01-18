from itertools import islice
import re
import warnings
from io import StringIO, UnsupportedOperation
import json
import numpy as np
import numbers
from ase.atoms import Atoms
from ase.calculators.calculator import all_properties, Calculator
from ase.calculators.singlepoint import SinglePointCalculator
from ase.spacegroup.spacegroup import Spacegroup
from ase.parallel import paropen
from ase.constraints import FixAtoms, FixCartesian
from ase.io.formats import index2range
from ase.utils import reader
class XYZChunk:

    def __init__(self, lines, natoms):
        self.lines = lines
        self.natoms = natoms

    def build(self):
        """Convert unprocessed chunk into Atoms."""
        return _read_xyz_frame(iter(self.lines), self.natoms)