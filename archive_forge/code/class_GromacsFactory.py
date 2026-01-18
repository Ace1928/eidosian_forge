import os
import re
from pathlib import Path
from typing import Mapping
import configparser
import pytest
from ase.calculators.calculator import (names as calculator_names,
@factory('gromacs')
class GromacsFactory:

    def __init__(self, executable):
        self.executable = executable

    def version(self):
        from ase.calculators.gromacs import get_gromacs_version
        return get_gromacs_version(self.executable)

    def calc(self, **kwargs):
        from ase.calculators.gromacs import Gromacs
        return Gromacs(command=self.executable, **kwargs)

    @classmethod
    def fromconfig(cls, config):
        return cls(config.executables['gromacs'])