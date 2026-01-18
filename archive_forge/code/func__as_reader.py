from __future__ import annotations
import logging
import os.path
import warnings
from typing import TYPE_CHECKING
import numpy as np
from monty.collections import AttrDict
from monty.dev import requires
from monty.functools import lazy_property
from monty.string import marquee
from pymatgen.core.structure import Structure
from pymatgen.core.units import ArrayWithUnit
from pymatgen.core.xcfunc import XcFunc
def _as_reader(file, cls):
    close_it = False
    if not isinstance(file, cls):
        file, close_it = (cls(file), True)
    return (file, close_it)