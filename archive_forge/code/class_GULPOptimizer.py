import os
import re
import numpy as np
from ase.units import eV, Ang
from ase.calculators.calculator import FileIOCalculator, ReadError
class GULPOptimizer:

    def __init__(self, atoms, calc):
        self.atoms = atoms
        self.calc = calc

    def todict(self):
        return {'type': 'optimization', 'optimizer': 'GULPOptimizer'}

    def run(self, fmax=None, steps=None, **gulp_kwargs):
        if fmax is not None:
            gulp_kwargs['gmax'] = fmax
        if steps is not None:
            gulp_kwargs['maxcyc'] = steps
        self.calc.set(**gulp_kwargs)
        self.atoms.calc = self.calc
        self.atoms.get_potential_energy()
        self.atoms.cell = self.calc.get_atoms().cell
        self.atoms.positions[:] = self.calc.get_atoms().positions