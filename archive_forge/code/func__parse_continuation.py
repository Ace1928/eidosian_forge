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
def _parse_continuation(self, value):
    if value is None:
        return None
    try:
        if self._options['reuse'].value:
            warnings.warn('Cannot set reuse if continuation is set, and vice versa. Set the other to None, if you want this setting.')
            return None
    except KeyError:
        pass
    return 'default' if value is True else str(value)