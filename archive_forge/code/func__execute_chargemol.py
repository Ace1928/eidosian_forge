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
def _execute_chargemol(self, **job_control_kwargs):
    """Internal function to run Chargemol.

        Args:
            atomic_densities_path (str): Path to the atomic densities directory
            required by Chargemol. If None, Pymatgen assumes that this is
            defined in a "DDEC6_ATOMIC_DENSITIES_DIR" environment variable.
                Default: None.
            job_control_kwargs: Keyword arguments for _write_jobscript_for_chargemol.
        """
    with ScratchDir('.'):
        try:
            os.symlink(self._chgcar_path, './CHGCAR')
            os.symlink(self._potcar_path, './POTCAR')
            os.symlink(self._aeccar0_path, './AECCAR0')
            os.symlink(self._aeccar2_path, './AECCAR2')
        except OSError as exc:
            print(f'Error creating symbolic link: {exc}')
        self._write_jobscript_for_chargemol(**job_control_kwargs)
        with subprocess.Popen(CHARGEMOL_EXE, stdout=subprocess.PIPE, stdin=subprocess.PIPE, close_fds=True) as rs:
            _stdout, stderr = rs.communicate()
        if rs.returncode != 0:
            raise RuntimeError(f'{CHARGEMOL_EXE} exit code: {rs.returncode}, error message: {stderr!s}. Please check your Chargemol installation.')
        self._from_data_dir()