import os
import warnings
import shutil
from os.path import join, isfile, islink
from typing import List, Sequence, Tuple
import numpy as np
import ase
from ase.calculators.calculator import kpts2ndarray
from ase.calculators.vasp.setups import get_default_setups
def _args_without_comment(data, marks=['!', '#']):
    """Check split arguments list for a comment, return data up to marker

    INCAR reader splits list arguments on spaces and leaves comment markers as
    individual items. This function returns only the data portion of the list.

    """
    comment_locs = [data.index(mark) for mark in marks if mark in data]
    if comment_locs == []:
        return data
    else:
        return data[:min(comment_locs)]