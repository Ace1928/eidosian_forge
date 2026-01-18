import os
import copy
import subprocess
from math import pi, sqrt
import pathlib
from typing import Union, Optional, List, Set, Dict, Any
import warnings
import numpy as np
from ase.cell import Cell
from ase.outputs import Properties, all_outputs
from ase.utils import jsonable
from ase.calculators.abc import GetPropertiesMixin
class FileIOCalculator(Calculator):
    """Base class for calculators that write/read input/output files."""
    command: Optional[str] = None
    'Command used to start calculation'

    def __init__(self, restart=None, ignore_bad_restart_file=Calculator._deprecated, label=None, atoms=None, command=None, **kwargs):
        """File-IO calculator.

        command: str
            Command used to start calculation.
        """
        Calculator.__init__(self, restart, ignore_bad_restart_file, label, atoms, **kwargs)
        if command is not None:
            self.command = command
        else:
            name = 'ASE_' + self.name.upper() + '_COMMAND'
            self.command = os.environ.get(name, self.command)

    def calculate(self, atoms=None, properties=['energy'], system_changes=all_changes):
        Calculator.calculate(self, atoms, properties, system_changes)
        self.write_input(self.atoms, properties, system_changes)
        if self.command is None:
            raise CalculatorSetupError('Please set ${} environment variable '.format('ASE_' + self.name.upper() + '_COMMAND') + 'or supply the command keyword')
        command = self.command
        if 'PREFIX' in command:
            command = command.replace('PREFIX', self.prefix)
        try:
            proc = subprocess.Popen(command, shell=True, cwd=self.directory)
        except OSError as err:
            msg = 'Failed to execute "{}"'.format(command)
            raise EnvironmentError(msg) from err
        errorcode = proc.wait()
        if errorcode:
            path = os.path.abspath(self.directory)
            msg = 'Calculator "{}" failed with command "{}" failed in {} with error code {}'.format(self.name, command, path, errorcode)
            raise CalculationFailed(msg)
        self.read_results()

    def write_input(self, atoms, properties=None, system_changes=None):
        """Write input file(s).

        Call this method first in subclasses so that directories are
        created automatically."""
        absdir = os.path.abspath(self.directory)
        if absdir != os.curdir and (not os.path.isdir(self.directory)):
            os.makedirs(self.directory)

    def read_results(self):
        """Read energy, forces, ... from output file(s)."""
        pass