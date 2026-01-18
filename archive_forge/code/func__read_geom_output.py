from copy import deepcopy
from os.path import isfile
from warnings import warn
from numpy import array
from ase import Atoms
from ase.calculators.calculator import FileIOCalculator, ReadError
from ase.parallel import paropen
from ase.units import Bohr, Hartree
def _read_geom_output(self, out):
    """Reads geometry optimisation output from ONETEP output file"""
    conv_fac = Bohr
    while 'x-----' not in out.readline():
        pass
    symbols = []
    positions = []
    line = out.readline()
    while 'xxxxxx' not in line:
        line = line.strip()
        pos = line.split()[3:6]
        pos = [conv_fac * float(p) for p in pos]
        atom = line.split()[1]
        positions.append(pos)
        symbols.append(atom)
        line = out.readline()
    if len(positions) != len(self.atoms):
        raise ReadError('Wrong number of atoms found in output geometryblock')
    if len(symbols) != len(self.atoms):
        raise ReadError('Wrong number of atoms found in output geometryblock')
    self.atoms.set_positions(positions)
    self.atoms.set_chemical_symbols(symbols)