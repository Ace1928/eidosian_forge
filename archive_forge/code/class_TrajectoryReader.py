import warnings
from typing import Tuple
import numpy as np
from ase import __version__
from ase.calculators.singlepoint import SinglePointCalculator, all_properties
from ase.constraints import dict2constraint
from ase.calculators.calculator import PropertyNotImplementedError
from ase.atoms import Atoms
from ase.io.jsonio import encode, decode
from ase.io.pickletrajectory import PickleTrajectory
from ase.parallel import world
from ase.utils import tokenize_version
class TrajectoryReader:
    """Reads Atoms objects from a .traj file."""

    def __init__(self, filename):
        """A Trajectory in read mode.

        The filename traditionally ends in .traj.
        """
        self.numbers = None
        self.pbc = None
        self.masses = None
        self._open(filename)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, tb):
        self.close()

    def _open(self, filename):
        import ase.io.ulm as ulm
        self.backend = ulm.open(filename, 'r')
        self._read_header()

    def _read_header(self):
        b = self.backend
        if b.get_tag() != 'ASE-Trajectory':
            raise IOError('This is not a trajectory file!')
        if len(b) > 0:
            self.pbc = b.pbc
            self.numbers = b.numbers
            self.masses = b.get('masses')
            self.constraints = b.get('constraints', '[]')
            self.description = b.get('description')
            self.version = b.version
            self.ase_version = b.get('ase_version')

    def close(self):
        """Close the trajectory file."""
        self.backend.close()

    def __getitem__(self, i=-1):
        if isinstance(i, slice):
            return SlicedTrajectory(self, i)
        b = self.backend[i]
        if 'numbers' in b:
            atoms = read_atoms(b, traj=self)
        else:
            atoms = read_atoms(b, header=[self.pbc, self.numbers, self.masses, self.constraints], traj=self)
        if 'calculator' in b:
            results = {}
            implemented_properties = []
            c = b.calculator
            for prop in all_properties:
                if prop in c:
                    results[prop] = c.get(prop)
                    implemented_properties.append(prop)
            calc = SinglePointCalculator(atoms, **results)
            calc.name = b.calculator.name
            calc.implemented_properties = implemented_properties
            if 'parameters' in c:
                calc.parameters.update(c.parameters)
            atoms.calc = calc
        return atoms

    def __len__(self):
        return len(self.backend)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]