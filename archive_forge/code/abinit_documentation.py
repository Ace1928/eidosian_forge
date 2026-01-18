import re
import ase.io.abinit as io
from ase.calculators.calculator import FileIOCalculator
from subprocess import check_output
Read results from ABINIT's text-output file.