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
@staticmethod
def _parse_int_vector(value):
    if isinstance(value, str):
        if ',' in value:
            value = value.replace(',', ' ')
        value = list(map(int, value.split()))
    value = np.array(value)
    if value.shape != (3,) or value.dtype != int:
        raise ValueError()
    return list(value)