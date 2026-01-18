from __future__ import annotations
import os
import shutil
import subprocess
import warnings
from datetime import datetime
from glob import glob
from pathlib import Path
from shutil import which
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any
import numpy as np
from monty.dev import deprecated
from monty.shutil import decompress_file
from monty.tempfile import ScratchDir
from pymatgen.io.common import VolumetricData
from pymatgen.io.vasp.inputs import Potcar
from pymatgen.io.vasp.outputs import Chgcar
def _parse_acf(self) -> list[dict]:
    """Parse Bader output file ACF.dat."""
    with open('ACF.dat', encoding='us-ascii') as file:
        lines = file.readlines()
    headers = ('x', 'y', 'z', 'charge', 'min_dist', 'atomic_vol')
    lines.pop(0)
    lines.pop(0)
    data = []
    while True:
        line = lines.pop(0).strip()
        if line.startswith('-'):
            break
        vals = map(float, line.split()[1:])
        data.append(dict(zip(headers, vals)))
    for line in lines:
        tokens = line.strip().split(':')
        if tokens[0] == 'VACUUM CHARGE':
            self.vacuum_charge = float(tokens[1])
        elif tokens[0] == 'VACUUM VOLUME':
            self.vacuum_volume = float(tokens[1])
        elif tokens[0] == 'NUMBER OF ELECTRONS':
            self.nelectrons = float(tokens[1])
    return data