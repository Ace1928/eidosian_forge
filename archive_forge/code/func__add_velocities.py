import os
import warnings
import numpy as np
import ase
from ase.data import atomic_masses
from ase.geometry import cellpar_to_cell
import collections
from functools import reduce
def _add_velocities(self):
    if not self._has_variable(self._velocities_var):
        self.nc.createVariable(self._velocities_var, 'f4', (self._frame_dim, self._atom_dim, self._spatial_dim))
        self.nc.variables[self._positions_var].units = 'Angstrom/Femtosecond'
        self.nc.variables[self._positions_var].scale_factor = 1.0