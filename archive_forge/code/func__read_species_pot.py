from copy import deepcopy
from os.path import isfile
from warnings import warn
from numpy import array
from ase import Atoms
from ase.calculators.calculator import FileIOCalculator, ReadError
from ase.parallel import paropen
from ase.units import Bohr, Hartree
def _read_species_pot(self, out):
    """ Read in pseudopotential information from a onetep output file"""
    line = out.readline().strip()
    pots = []
    while '%endblock' not in line.lower() and len(line) > 0:
        atom, suffix = line.split(None, 1)
        filename = suffix.split('#', 1)[0].strip()
        filename = filename.replace('"', '')
        filename = filename.replace("'", '')
        pots.append((atom, filename))
        line = out.readline().strip()
    if len(line) == 0:
        raise ReadError('End of file while reading potential block')
    self.set_pseudos(pots)