from __future__ import annotations
import os
import subprocess
import warnings
from glob import glob
from shutil import which
from typing import TYPE_CHECKING
import numpy as np
from monty.tempfile import ScratchDir
from pymatgen.core import Element
from pymatgen.io.vasp.inputs import Potcar
from pymatgen.io.vasp.outputs import Chgcar
@staticmethod
def _get_filepath(path, filename, suffix=''):
    """Returns the full path to the filename in the path. Works even if the file has
        a .gz extension.

        Args:
            path (str): Path to the file.
            filename (str): Filename.
            suffix (str): Optional suffix at the end of the filename.

        Returns:
            str: Absolute path to the file.
        """
    name_pattern = f'{filename}{suffix}*' if filename != 'POTCAR' else f'{filename}*'
    paths = glob(os.path.join(path, name_pattern))
    fpath = None
    if len(paths) >= 1:
        paths.sort(reverse=True)
        if len(paths) > 1:
            warnings.warn(f'Multiple files detected, using {os.path.basename(paths[0])}')
        fpath = paths[0]
    return fpath