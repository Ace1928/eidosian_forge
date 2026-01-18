from collections.abc import Mapping, Sequence
from subprocess import check_call, DEVNULL
from os import unlink
from pathlib import Path
import numpy as np
from ase.io.utils import PlottingVariables
from ase.constraints import FixAtoms
from ase import Atoms
class POVRAYInputs:

    def __init__(self, path):
        self.path = path

    def render(self, povray_executable='povray', stderr=DEVNULL, clean_up=False):
        cmd = [povray_executable, str(self.path)]
        check_call(cmd, stderr=stderr)
        png_path = self.path.with_suffix('.png').absolute()
        if not png_path.is_file():
            raise RuntimeError(f'Povray left no output PNG file "{png_path}"')
        if clean_up:
            unlink(self.path)
            unlink(self.path.with_suffix('.pov'))
        return png_path