import subprocess
import numpy as np
from ase.calculators.calculator import Calculator, FileIOCalculator
import ase.units as units
from scipy.io import netcdf
 Modify amber topology charges to contain the updated
            QM charges, needed in QM/MM.
            Using amber's parmed program to change charges.
        