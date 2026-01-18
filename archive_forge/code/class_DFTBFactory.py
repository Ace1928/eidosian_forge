import os
import re
from pathlib import Path
from typing import Mapping
import configparser
import pytest
from ase.calculators.calculator import (names as calculator_names,
@factory('dftb')
class DFTBFactory:

    def __init__(self, executable, skt_paths):
        self.executable = executable
        assert len(skt_paths) == 1
        self.skt_path = skt_paths[0]

    def version(self):
        stdout = read_stdout([self.executable])
        match = re.search('DFTB\\+ release\\s*(\\S+)', stdout, re.M)
        return match.group(1)

    def calc(self, **kwargs):
        from ase.calculators.dftb import Dftb
        command = f'{self.executable} > PREFIX.out'
        return Dftb(command=command, slako_dir=str(self.skt_path) + '/', **kwargs)

    @classmethod
    def fromconfig(cls, config):
        return cls(config.executables['dftb'], config.datafiles['dftb'])