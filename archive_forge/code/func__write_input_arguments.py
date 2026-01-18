import os
import os.path as op
import subprocess
import shutil
import numpy as np
from ase.units import Bohr, Hartree
import ase.data
from ase.calculators.calculator import FileIOCalculator, ReadError
from ase.calculators.calculator import Parameters, all_changes
from ase.calculators.calculator import equal
import ase.io
from .demon_io import parse_xray
def _write_input_arguments(self, fd):
    """Write directly given input-arguments."""
    input_arguments = self.parameters['input_arguments']
    if input_arguments is None:
        return
    for key, value in input_arguments.items():
        self._write_argument(key, value, fd)