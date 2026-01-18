from copy import deepcopy
from os.path import isfile
from warnings import warn
from numpy import array
from ase import Atoms
from ase.calculators.calculator import FileIOCalculator, ReadError
from ase.parallel import paropen
from ase.units import Bohr, Hartree
def _read_species_solver(self, out, cond=False):
    """ Read in pseudopotential information from a onetep output file"""
    line = out.readline().strip()
    solvers = []
    while '%endblock' not in line.lower() and len(line) > 0:
        atom, suffix = line.split(None, 1)
        solver_str = suffix.split('#', 1)[0].strip()
        solvers.append((atom, solver_str))
        line = out.readline().strip()
    if len(line) == 0:
        raise ReadError('End of file while reading solver block')
    if not cond:
        self.set_solvers(solvers)
    else:
        self.set_solvers_cond(solvers)