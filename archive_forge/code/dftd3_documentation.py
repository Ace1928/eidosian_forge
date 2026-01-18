import os
import subprocess
from warnings import warn
import numpy as np
from ase.calculators.calculator import (Calculator, FileIOCalculator,
from ase.io import write
from ase.io.vasp import write_vasp
from ase.parallel import world
from ase.units import Bohr, Hartree
Grimme DFT-D3 calculator