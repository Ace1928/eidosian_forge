import os
import re
from pathlib import Path
from typing import Mapping
import configparser
import pytest
from ase.calculators.calculator import (names as calculator_names,
@factory('castep')
class CastepFactory:

    def __init__(self, executable):
        self.executable = executable

    def version(self):
        from ase.calculators.castep import get_castep_version
        return get_castep_version(self.executable)

    def calc(self, **kwargs):
        from ase.calculators.castep import Castep
        return Castep(castep_command=self.executable, **kwargs)

    @classmethod
    def fromconfig(cls, config):
        return cls(config.executables['castep'])