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
def _parse_symmetry_ops(self, value):
    if not isinstance(value, tuple) or not len(value) == 2 or (not value[0].shape[1:] == (3, 3)) or (not value[1].shape[1:] == (3,)) or (not value[0].shape[0] == value[1].shape[0]):
        warnings.warn('Invalid symmetry_ops block, skipping')
        return
    text_block = ''
    for op_i, (op_rot, op_tranls) in enumerate(zip(*value)):
        text_block += '\n'.join([' '.join([str(x) for x in row]) for row in op_rot])
        text_block += '\n'
        text_block += ' '.join([str(x) for x in op_tranls])
        text_block += '\n\n'
    return text_block