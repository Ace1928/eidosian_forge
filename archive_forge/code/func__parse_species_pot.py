import difflib
import numpy as np
import os
import re
import glob
import shutil
import sys
import json
import time
import tempfile
import warnings
import subprocess
from copy import deepcopy
from collections import namedtuple
from itertools import product
from typing import List, Set
import ase
import ase.units as units
from ase.calculators.general import Calculator
from ase.calculators.calculator import compare_atoms
from ase.calculators.calculator import PropertyNotImplementedError
from ase.calculators.calculator import kpts2sizeandoffsets
from ase.dft.kpoints import BandPath
from ase.parallel import paropen
from ase.io.castep import read_param
from ase.io.castep import read_bands
from ase.constraints import FixCartesian
def _parse_species_pot(self, value):
    if isinstance(value, tuple) and len(value) == 2:
        value = [value]
    if hasattr(value, '__getitem__'):
        pspots = [tuple(map(str.strip, x)) for x in value]
        if not all(map(lambda x: len(x) == 2, value)):
            warnings.warn('Please specify pseudopotentials in python as a tuple or a list of tuples formatted like: (species, file), e.g. ("O", "path-to/O_OTFG.usp") Anything else will be ignored')
            return None
    text_block = self._options['species_pot'].value
    text_block = text_block if text_block else ''
    for pp in pspots:
        text_block = re.sub('\\n?\\s*%s\\s+.*' % pp[0], '', text_block)
        if pp[1]:
            text_block += '\n%s %s' % pp
    return text_block