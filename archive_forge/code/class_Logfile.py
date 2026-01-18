import sys
import time
import copy
import warnings
from abc import ABC, abstractmethod
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.interpolate import CubicSpline
from ase.constraints import Filter, FixAtoms
from ase.utils import longsum
from ase.geometry import find_mic
import ase.utils.ff as ff
import ase.units as units
from ase.optimize.precon.neighbors import (get_neighbours,
from ase.neighborlist import neighbor_list
class Logfile:

    def __init__(self, logfile=None):
        if isinstance(logfile, str):
            if logfile == '-':
                logfile = sys.stdout
            else:
                logfile = open(logfile, 'a')
        self.logfile = logfile

    def write(self, *args):
        if self.logfile is None:
            return
        self.logfile.write(*args)