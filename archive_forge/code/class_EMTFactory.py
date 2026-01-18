import os
import re
from pathlib import Path
from typing import Mapping
import configparser
import pytest
from ase.calculators.calculator import (names as calculator_names,
@factory('emt')
class EMTFactory(BuiltinCalculatorFactory):
    pass