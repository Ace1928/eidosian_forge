import os
import os.path as op
import pathlib as pl
import numpy as np
from ase.units import Bohr, Hartree
import ase.data
from ase.calculators.calculator import FileIOCalculator, ReadError
from ase.calculators.calculator import Parameters
import ase.io
class DemonNanoParameters(Parameters):
    """Parameters class for the calculator.

    The options here are the most important ones that the user needs to be
    aware of. Further options accepted by deMon can be set in the dictionary
    input_arguments.

    """

    def __init__(self, label='.', atoms=None, command=None, basis_path=None, restart_path='.', print_out='ASE', title='deMonNano input file', forces=False, input_arguments=None):
        kwargs = locals()
        kwargs.pop('self')
        Parameters.__init__(self, **kwargs)