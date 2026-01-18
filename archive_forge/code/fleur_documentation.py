import os
from subprocess import Popen, PIPE
import re
import numpy as np
from ase.units import Hartree, Bohr
from ase.calculators.calculator import PropertyNotImplementedError
Check the convergence of calculation