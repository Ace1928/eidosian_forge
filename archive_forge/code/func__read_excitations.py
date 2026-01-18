from copy import deepcopy
from os.path import isfile
from warnings import warn
from numpy import array
from ase import Atoms
from ase.calculators.calculator import FileIOCalculator, ReadError
from ase.parallel import paropen
from ase.units import Bohr, Hartree
def _read_excitations(self, out):
    """ Extract the computed electronic excitations from a onetep output
        file."""
    excitations = []
    line = out.readline()
    while line:
        words = line.split()
        if len(words) == 0:
            break
        excitations.append([float(words[0]), float(words[1]) * Hartree, float(words[2])])
        line = out.readline()
    self.results['excitations'] = array(excitations)