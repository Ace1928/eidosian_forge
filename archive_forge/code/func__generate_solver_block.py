from copy import deepcopy
from os.path import isfile
from warnings import warn
from numpy import array
from ase import Atoms
from ase.calculators.calculator import FileIOCalculator, ReadError
from ase.parallel import paropen
from ase.units import Bohr, Hartree
def _generate_solver_block(self, cond=False):
    """Create a default onetep pseudoatomic solvers block, using 'SOLVE'
        unless the user has set overrides for specific species by setting
        specific entries in species_solver (_cond)"""
    if not cond:
        solver_var = 'species_solver'
    else:
        solver_var = 'species_solver_cond'
    for sp in self.species:
        try:
            atomic_string = self.parameters[solver_var][sp[0]]
        except KeyError:
            atomic_string = 'SOLVE'
        if not cond:
            self.solvers.append((sp[0], atomic_string))
        else:
            self.solvers_cond.append((sp[0], atomic_string))