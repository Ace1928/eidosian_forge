import os
import re
import warnings
from subprocess import Popen, PIPE
from math import log10, floor
import numpy as np
from ase import Atoms
from ase.units import Ha, Bohr
from ase.io import read, write
from ase.calculators.calculator import FileIOCalculator
from ase.calculators.calculator import PropertyNotImplementedError, ReadError
def add_data_group(data_group, string=None, raw=False):
    """write a turbomole data group to control file"""
    if raw:
        data = data_group
    else:
        data = '$' + data_group
        if string:
            data += ' ' + string
        data += '\n'
    fd = open('control', 'r+')
    lines = fd.readlines()
    fd.seek(0)
    fd.truncate()
    lines.insert(2, data)
    fd.write(''.join(lines))
    fd.close()