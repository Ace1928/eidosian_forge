from __future__ import annotations
import datetime
import itertools
import logging
import math
import os
import re
import warnings
import xml.etree.ElementTree as ET
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from glob import glob
from io import StringIO
from pathlib import Path
from typing import TYPE_CHECKING, Literal
import numpy as np
from monty.io import reverse_readfile, zopen
from monty.json import MSONable, jsanitize
from monty.os.path import zpath
from monty.re import regrep
from numpy.testing import assert_allclose
from pymatgen.core import Composition, Element, Lattice, Structure
from pymatgen.core.units import unitized
from pymatgen.electronic_structure.bandstructure import (
from pymatgen.electronic_structure.core import Magmom, Orbital, OrbitalType, Spin
from pymatgen.electronic_structure.dos import CompleteDos, Dos
from pymatgen.entries.computed_entries import ComputedEntry, ComputedStructureEntry
from pymatgen.io.common import VolumetricData as BaseVolumetricData
from pymatgen.io.core import ParseError
from pymatgen.io.vasp.inputs import Incar, Kpoints, Poscar, Potcar
from pymatgen.io.wannier90 import Unk
from pymatgen.util.io_utils import clean_lines, micro_pyawk
from pymatgen.util.num import make_symmetric_matrix_from_upper_tri
def _parse_v_parameters(val_type, val, filename, param_name):
    """
    Helper function to convert a Vasprun array-type parameter into the proper
    type. Boolean, int and float types are converted.

    Args:
        val_type: Value type parsed from vasprun.xml.
        val: Actual string value parsed for vasprun.xml.
        filename: Fullpath of vasprun.xml. Used for robust error handling.
            E.g., if vasprun.xml contains *** for some Incar parameters,
            the code will try to read from an INCAR file present in the same
            directory.
        param_name: Name of parameter.

    Returns:
        Parsed value.
    """
    err = ValueError('Error in parsing vasprun.xml')
    if val_type == 'logical':
        val = [i == 'T' for i in val.split()]
    elif val_type == 'int':
        try:
            val = [int(i) for i in val.split()]
        except ValueError:
            val = _parse_from_incar(filename, param_name)
            if val is None:
                raise err
    elif val_type == 'string':
        val = val.split()
    else:
        try:
            val = [float(i) for i in val.split()]
        except ValueError:
            val = _parse_from_incar(filename, param_name)
            if val is None:
                raise err
    return val