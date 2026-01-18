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
def _set_restart(self, params_update):
    """constructs atoms, parameters and results from a previous
        calculation"""
    self.read_restart()
    params_old = self.read_parameters()
    for p in list(params_update.keys()):
        if not self.parameter_updateable[p]:
            del params_update[p]
            warnings.warn('"' + p + '"' + ' cannot be changed')
    params_new = params_old.copy()
    params_new.update(params_update)
    self.set_parameters(params_new)
    self.verify_parameters()
    if self.define_str:
        execute('define', input_str=self.define_str)
    if params_update or self.control_kdg or self.control_input:
        self._update_data_groups(params_old, params_update)
    self.initialized = True
    self.update_energy = True
    self.update_forces = True
    self.update_geometry = True
    self.update_hessian = True