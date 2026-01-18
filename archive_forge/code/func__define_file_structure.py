import os
import warnings
import numpy as np
import ase
from ase.data import atomic_masses
from ase.geometry import cellpar_to_cell
import collections
from functools import reduce
def _define_file_structure(self, atoms):
    self.nc.Conventions = 'AMBER'
    self.nc.ConventionVersion = '1.0'
    self.nc.program = 'ASE'
    self.nc.programVersion = ase.__version__
    self.nc.title = 'MOL'
    if self._frame_dim not in self.nc.dimensions:
        self.nc.createDimension(self._frame_dim, None)
    if self._spatial_dim not in self.nc.dimensions:
        self.nc.createDimension(self._spatial_dim, 3)
    if self._atom_dim not in self.nc.dimensions:
        self.nc.createDimension(self._atom_dim, len(atoms))
    if self._cell_spatial_dim not in self.nc.dimensions:
        self.nc.createDimension(self._cell_spatial_dim, 3)
    if self._cell_angular_dim not in self.nc.dimensions:
        self.nc.createDimension(self._cell_angular_dim, 3)
    if self._label_dim not in self.nc.dimensions:
        self.nc.createDimension(self._label_dim, 5)
    if not self._has_variable(self._spatial_var):
        self.nc.createVariable(self._spatial_var, 'S1', (self._spatial_dim,))
        self.nc.variables[self._spatial_var][:] = ['x', 'y', 'z']
    if not self._has_variable(self._cell_spatial_var):
        self.nc.createVariable(self._cell_spatial_dim, 'S1', (self._cell_spatial_dim,))
        self.nc.variables[self._cell_spatial_var][:] = ['a', 'b', 'c']
    if not self._has_variable(self._cell_angular_var):
        self.nc.createVariable(self._cell_angular_var, 'S1', (self._cell_angular_dim, self._label_dim))
        self.nc.variables[self._cell_angular_var][0] = [x for x in 'alpha']
        self.nc.variables[self._cell_angular_var][1] = [x for x in 'beta ']
        self.nc.variables[self._cell_angular_var][2] = [x for x in 'gamma']
    if not self._has_variable(self._numbers_var):
        self.nc.createVariable(self._numbers_var[0], 'i', (self._frame_dim, self._atom_dim))
    if not self._has_variable(self._positions_var):
        self.nc.createVariable(self._positions_var, 'f4', (self._frame_dim, self._atom_dim, self._spatial_dim))
        self.nc.variables[self._positions_var].units = 'Angstrom'
        self.nc.variables[self._positions_var].scale_factor = 1.0
    if not self._has_variable(self._cell_lengths_var):
        self.nc.createVariable(self._cell_lengths_var, 'd', (self._frame_dim, self._cell_spatial_dim))
        self.nc.variables[self._cell_lengths_var].units = 'Angstrom'
        self.nc.variables[self._cell_lengths_var].scale_factor = 1.0
    if not self._has_variable(self._cell_angles_var):
        self.nc.createVariable(self._cell_angles_var, 'd', (self._frame_dim, self._cell_angular_dim))
        self.nc.variables[self._cell_angles_var].units = 'degree'
    if not self._has_variable(self._cell_origin_var):
        self.nc.createVariable(self._cell_origin_var, 'd', (self._frame_dim, self._cell_spatial_dim))
        self.nc.variables[self._cell_origin_var].units = 'Angstrom'
        self.nc.variables[self._cell_origin_var].scale_factor = 1.0