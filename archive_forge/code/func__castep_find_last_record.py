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
def _castep_find_last_record(self, castep_file):
    """Checks wether a given castep file has a regular
        ending message following the last banner message. If this
        is the case, the line number of the last banner is message
        is return, otherwise False.

        returns (record_start, record_end, end_found, last_record_complete)
        """
    if isinstance(castep_file, str):
        castep_file = paropen(castep_file, 'r')
        file_opened = True
    else:
        file_opened = False
    record_starts = []
    while True:
        line = castep_file.readline()
        if 'Welcome' in line and 'CASTEP' in line:
            record_starts = [castep_file.tell()] + record_starts
        if not line:
            break
    if record_starts == []:
        warnings.warn('Could not find CASTEP label in result file: %s. Are you sure this is a .castep file?' % castep_file)
        return
    end_found = False
    record_end = -1
    for record_nr, record_start in enumerate(record_starts):
        castep_file.seek(record_start)
        while True:
            line = castep_file.readline()
            if not line:
                break
            if 'warn' in line.lower():
                self._warnings.append(line)
            if 'Finalisation time   =' in line:
                end_found = True
                record_end = castep_file.tell()
                break
        if end_found:
            break
    if file_opened:
        castep_file.close()
    if end_found:
        if record_nr == 0:
            return (record_start, record_end, True, True)
        else:
            return (record_start, record_end, True, False)
    else:
        return (0, record_end, False, False)