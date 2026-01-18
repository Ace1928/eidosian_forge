import os
import subprocess
from glob import glob
from shutil import which
import numpy as np
from ase import units
from ase.calculators.calculator import (EnvironmentError,
from ase.io.gromos import read_gromos, write_gromos
def _execute_gromacs(self, command):
    """ execute gmx command
        Parameters
        ----------
        command : str
        """
    if self.command:
        subprocess.check_call(self.command + ' ' + command, shell=True)
    else:
        raise EnvironmentError(self.missing_gmx)