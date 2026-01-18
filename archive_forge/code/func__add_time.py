import os
import warnings
import numpy as np
import ase
from ase.data import atomic_masses
from ase.geometry import cellpar_to_cell
import collections
from functools import reduce
def _add_time(self):
    if not self._has_variable(self._time_var):
        self.nc.createVariable(self._time_var, 'f8', (self._frame_dim,))