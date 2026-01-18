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
def _add_command(self, stage_name: str, command: str, args: str | float | None=None) -> None:
    """
        Helper method to add a single LAMMPS command and its arguments to
        the LammpsInputFile. The stage name should be provided: a default behavior
        is avoided here to avoid mistakes.

        Example:
            In order to add the command ``pair_coeff 1 1 morse 0.0580 3.987 3.404``
            to the stage "Definition of the potential", simply use
            ```
            your_input_file._add_command(
                stage_name="Definition of the potential",
                command="pair_coeff 1 1 morse 0.0580 3.987 3.404"
            )
            ```
            or
            ```
            your_input_file._add_command(
                stage_name="Definition of the potential",
                command="pair_coeff",
                args="1 1 morse 0.0580 3.987 3.404"
            )
            ```

        Args:
            stage_name (str): name of the stage to which the command should be added.
            command (str): LAMMPS command, with or without the arguments.
            args (str): Arguments for the LAMMPS command.
        """
    if args is None:
        string_split = command.split()
        command = string_split[0]
        args = ' '.join(string_split[1:])
    idx = self.stages_names.index(stage_name)
    if not self.stages[idx]['commands']:
        self.stages[idx]['commands'] = [(command, args)]
    else:
        self.stages[idx]['commands'].append((command, args))