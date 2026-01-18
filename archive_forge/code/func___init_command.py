import os
import warnings
import time
from typing import Optional
import re
import numpy as np
from ase.units import Hartree
from ase.io.aims import write_aims, read_aims
from ase.data import atomic_numbers
from ase.calculators.calculator import FileIOCalculator, Parameters, kpts2mp, \
def __init_command(self, command=None, aims_command=None, outfilename=None):
    """
        Create the private variables for which properties are defines and set
        them accordingly.
        """
    self.__aims_command = None
    self.__outfilename = None
    self.__command: Optional[str] = None
    self.__update_command(command=command, aims_command=aims_command, outfilename=outfilename)