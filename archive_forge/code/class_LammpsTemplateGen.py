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
class LammpsTemplateGen(TemplateInputGen):
    """
    Creates an InputSet object for a LAMMPS run based on a template file.
    The input script is constructed by substituting variables into placeholders
    in the template file using python's Template.safe_substitute() function.
    The data file containing coordinates and topology information can be provided
    as a LammpsData instance. Alternatively, you can include a read_data command
    in the template file that points to an existing data file.
    Other supporting files are not handled at the moment.

    To write the input files to a directory, call LammpsTemplateSet.write_input()
    See pymatgen.io.template.py for additional documentation of this method.
    """

    def get_input_set(self, script_template: str | Path, settings: dict | None=None, script_filename: str='in.lammps', data: LammpsData | CombinedData | None=None, data_filename: str='system.data') -> InputSet:
        """
        Args:
            script_template: String template for input script with
                placeholders. The format for placeholders has to be
                '$variable_name', e.g., '$temperature'
            settings: Contains values to be written to the
                placeholders, e.g., {'temperature': 1}. Default to None.
            data: Data file as a LammpsData instance. Default to None, i.e., no
                data file supplied. Note that a matching 'read_data' command
                must be provided in the script template in order for the data
                file to actually be read.
            script_filename: Filename for the input file.
            data_filename: Filename for the data file, if provided.
        """
        input_set = super().get_input_set(template=script_template, variables=settings, filename=script_filename)
        if data:
            input_set.update({data_filename: data})
        return input_set