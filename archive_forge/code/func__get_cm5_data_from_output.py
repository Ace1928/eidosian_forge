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
def _get_cm5_data_from_output(ddec_analysis_path):
    """Internal command to process Chargemol CM5 data.

        Args:
            ddec_analysis_path (str): Path VASP_DDEC_analysis.output file

        Returns:
            list[float]: CM5 charges
        """
    props = []
    if os.path.isfile(ddec_analysis_path):
        start = False
        with open(ddec_analysis_path) as r:
            for line in r:
                if 'computed CM5' in line:
                    start = True
                    continue
                if 'Hirshfeld and CM5' in line:
                    break
                if start:
                    vals = line.split()
                    props.extend([float(c) for c in [val.strip() for val in vals]])
    else:
        raise FileNotFoundError(f'{ddec_analysis_path} not found')
    return props