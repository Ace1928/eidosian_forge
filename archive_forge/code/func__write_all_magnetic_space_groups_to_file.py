from __future__ import annotations
import os
import sqlite3
import textwrap
from array import array
from fractions import Fraction
from typing import TYPE_CHECKING
import numpy as np
from monty.design_patterns import cached_class
from pymatgen.core.operations import MagSymmOp
from pymatgen.electronic_structure.core import Magmom
from pymatgen.symmetry.groups import SymmetryGroup, in_array_list
from pymatgen.symmetry.settings import JonesFaithfulTransformation
from pymatgen.util.string import transformation_to_string
def _write_all_magnetic_space_groups_to_file(filename):
    """Write all magnetic space groups to a human-readable text file.
    Should contain same information as text files provided by ISO-MAG.
    """
    out = 'Data parsed from raw data from:\nISO-MAG, ISOTROPY Software Suite, iso.byu.edu\nhttp://stokes.byu.edu/iso/magnetic_data.txt\nUsed with kind permission from Professor Branton Campbell, BYU\n\n'
    all_msgs = []
    for i in range(1, 1652):
        all_msgs.append(MagneticSpaceGroup(i))
    for msg in all_msgs:
        out += f'\n{msg.data_str()}\n\n--------\n'
    with open(filename, mode='w') as file:
        file.write(out)