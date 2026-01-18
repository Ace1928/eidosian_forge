import os
import re
from pathlib import Path
from typing import Mapping
import configparser
import pytest
from ase.calculators.calculator import (names as calculator_names,
class BuiltinCalculatorFactory:

    def calc(self, **kwargs):
        from ase.calculators.calculator import get_calculator_class
        cls = get_calculator_class(self.name)
        return cls(**kwargs)

    @classmethod
    def fromconfig(cls, config):
        return cls()