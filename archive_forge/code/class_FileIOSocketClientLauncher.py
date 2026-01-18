import os
import socket
from subprocess import Popen, PIPE
from contextlib import contextmanager
import numpy as np
from ase.calculators.calculator import (Calculator, all_changes,
import ase.units as units
from ase.utils import IOContext
from ase.stress import full_3x3_to_voigt_6_stress
class FileIOSocketClientLauncher:

    def __init__(self, calc):
        self.calc = calc

    def __call__(self, atoms, properties=None, port=None, unixsocket=None):
        assert self.calc is not None
        cmd = self.calc.command.replace('PREFIX', self.calc.prefix)
        self.calc.write_input(atoms, properties=properties, system_changes=all_changes)
        cwd = self.calc.directory
        cmd = cmd.format(port=port, unixsocket=unixsocket)
        return Popen(cmd, shell=True, cwd=cwd)