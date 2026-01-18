import os
import re
from pathlib import Path
from typing import Mapping
import configparser
import pytest
from ase.calculators.calculator import (names as calculator_names,
@factory('openmx')
class OpenMXFactory:

    def __init__(self, executable, data_path):
        self.executable = executable
        self.data_path = data_path

    def version(self):
        from ase.calculators.openmx.openmx import parse_omx_version
        dummyfile = 'omx_dummy_input'
        stdout = read_stdout([self.executable, dummyfile], createfile=dummyfile)
        return parse_omx_version(stdout)

    def calc(self, **kwargs):
        from ase.calculators.openmx import OpenMX
        return OpenMX(command=self.executable, data_path=str(self.data_path), **kwargs)

    @classmethod
    def fromconfig(cls, config):
        return cls(config.executables['openmx'], data_path=config.datafiles['openmx'][0])