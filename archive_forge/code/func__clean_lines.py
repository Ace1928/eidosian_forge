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
def _clean_lines(string_list: list, ignore_comments: bool=False) -> list:
    """
        Helper method to strips whitespaces, carriage returns and redundant empty
        lines from a list of strings.
        Transforms "& \\n" and "&\\n" into "" as the & symbol means the line continues.
        Also removes lines with "# LAMMPS input generated from LammpsInputFile"
        to avoid possible repetitions.

        Args:
            string_list (list): List of strings.
            ignore_comments (bool): True if the strings starting with # should be ignored.

        Returns:
            List of strings
        """
    if len(string_list) == 0 or all((s == '' for s in string_list)):
        raise ValueError('The list of strings should contain some non-empty strings.')
    while '# LAMMPS input generated from LammpsInputFile' in string_list:
        string_list.remove('# LAMMPS input generated from LammpsInputFile')
    imin = len(string_list)
    imax = 0
    for idx, string in enumerate(string_list):
        if string != '' and idx <= imin:
            imin = idx
        if string != '' and idx >= imax:
            imax = idx
    string_list = string_list[imin:imax + 1]
    new_list = []
    for string in string_list:
        if len(string) > 1 or not (len(string.strip()) == 1 and string[0] == '#'):
            new_list.append(string)
    string_list = new_list
    lines = [string_list[0]]
    for idx, string in enumerate(string_list[1:-1]):
        if string != '' and (not (string[0] == '#' and ignore_comments)) or (string == '' and string_list[idx + 2] != ''):
            lines.append(string)
    lines.append(string_list[-1])
    return lines