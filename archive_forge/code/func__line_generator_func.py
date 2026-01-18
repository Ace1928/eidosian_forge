import numpy as np
from ase.atoms import Atoms
from ase.units import Hartree
from ase.data import atomic_numbers
from ase.calculators.singlepoint import SinglePointCalculator
from ase.utils import writer, reader
def _line_generator_func():
    for line in fileobj:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        yield line