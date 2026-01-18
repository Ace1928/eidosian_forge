import os
import re
from pathlib import Path
from typing import Mapping
import configparser
import pytest
from ase.calculators.calculator import (names as calculator_names,
@factory('dftd3')
class DFTD3Factory:

    def __init__(self, executable):
        self.executable = executable

    def calc(self, **kwargs):
        from ase.calculators.dftd3 import DFTD3
        return DFTD3(command=self.executable, **kwargs)

    @classmethod
    def fromconfig(cls, config):
        return cls(config.executables['dftd3'])