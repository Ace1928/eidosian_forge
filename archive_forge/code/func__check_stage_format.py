from __future__ import annotations
import os
import re
import shutil
import warnings
from pathlib import Path
from string import Template
from typing import TYPE_CHECKING
import numpy as np
from monty.dev import deprecated
from monty.io import zopen
from monty.json import MSONable
from pymatgen.core import __version__ as CURRENT_VER
from pymatgen.io.core import InputFile
from pymatgen.io.lammps.data import CombinedData, LammpsData
from pymatgen.io.template import TemplateInputGen
@staticmethod
def _check_stage_format(stage: dict) -> None:
    if list(stage) != ['stage_name', 'commands']:
        raise KeyError("The provided stage does not have the correct keys. It should be 'stage_name' and 'commands'.")
    if not isinstance(stage['stage_name'], str):
        raise TypeError("The value of 'stage_name' should be a string.")
    if not isinstance(stage['commands'], list):
        raise TypeError('The provided commands should be a list.')
    if len(stage['commands']) >= 1 and (not all((len(cmdargs) == 2 for cmdargs in stage['commands'])) or not all((isinstance(cmd, str) and isinstance(arg, str) for cmd, arg in stage['commands']))):
        raise ValueError('The provided commands should be a list of 2-strings tuples.')