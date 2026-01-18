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
def _parse_float_physical(value):
    if isinstance(value, str):
        value = value.split()
    try:
        l = len(value)
    except TypeError:
        l = 1
        value = [value]
    if l == 1:
        try:
            value = (float(value[0]), '')
        except (TypeError, ValueError):
            raise ValueError()
    elif l == 2:
        try:
            value = (float(value[0]), value[1])
        except (TypeError, ValueError, IndexError):
            raise ValueError()
    else:
        raise ValueError()
    return value