from copy import deepcopy
from os.path import isfile
from warnings import warn
from numpy import array
from ase import Atoms
from ase.calculators.calculator import FileIOCalculator, ReadError
from ase.parallel import paropen
from ase.units import Bohr, Hartree
def _read_forces(self, out):
    """ Extract the computed forces from a onetep output file"""
    forces = []
    atomic2ang = Hartree / Bohr
    while True:
        line = out.readline()
        fields = line.split()
        if len(fields) > 6:
            break
    while len(fields) == 7:
        force = [float(fcomp) * atomic2ang for fcomp in fields[-4:-1]]
        forces.append(force)
        line = out.readline()
        fields = line.split()
    self.results['forces'] = array(forces)