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
def _initialize_stage(self, stage_name: str | None=None, stage_index: int | None=None) -> None:
    """
        Initialize an empty stage with the given name in the LammpsInputFile.

        Args:
            stage_name (str): If a stage name is mentioned, the command is added
                under that stage block, else the new stage is named from numbering.
                If given, stage_name cannot be one of the already present stage names.
            stage_index (int): Index of the stage where it should be added.
                If None, the stage is added at the end of the LammpsInputFile.
        """
    if stage_name is None:
        stage_name = f'Stage {self.nstages + 1}'
        if stage_name in self.stages_names:
            stage_numbers = [int(stage_name.split()[1]) for stage_name in self.stages_names if stage_name.split()[0] == 'Stage' and stage_name.split()[1].isdigit()]
            stage_name = f'Stage {np.max(stage_numbers) + 1}'
    if not isinstance(stage_name, str):
        raise TypeError('Stage names should be strings.')
    if stage_name in self.stages_names:
        raise ValueError('The provided stage name is already present in LammpsInputFile.stages.')
    if stage_index is None or stage_index == -1:
        self.stages.append({'stage_name': stage_name, 'commands': []})
    elif stage_index > len(self.stages):
        raise IndexError('The provided index is too large with respect to the current number of stages.')
    else:
        self.stages.insert(stage_index, {'stage_name': stage_name, 'commands': []})