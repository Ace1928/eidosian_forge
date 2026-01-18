import os
import re
from pathlib import Path
from typing import Mapping
import configparser
import pytest
from ase.calculators.calculator import (names as calculator_names,
@factory('aims')
class AimsFactory:

    def __init__(self, executable):
        self.executable = executable

    def calc(self, **kwargs):
        from ase.calculators.aims import Aims
        kwargs1 = dict(xc='LDA')
        kwargs1.update(kwargs)
        return Aims(command=self.executable, **kwargs1)

    def version(self):
        from ase.calculators.aims import get_aims_version
        txt = read_stdout([self.executable])
        return get_aims_version(txt)

    @classmethod
    def fromconfig(cls, config):
        return cls(config.executables['aims'])