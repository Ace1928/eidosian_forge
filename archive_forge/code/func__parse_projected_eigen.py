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
@staticmethod
def _parse_projected_eigen(elem):
    root = elem.find('array').find('set')
    proj_eigen = defaultdict(list)
    for s in root.findall('set'):
        spin = int(re.match('spin(\\d+)', s.attrib['comment']).group(1))
        for ss in s.findall('set'):
            dk = []
            for sss in ss.findall('set'):
                db = _parse_vasp_array(sss)
                dk.append(db)
            proj_eigen[spin].append(dk)
    proj_eigen = {spin: np.array(v) for spin, v in proj_eigen.items()}
    if len(proj_eigen) > 2:
        proj_mag = np.stack([proj_eigen.pop(i) for i in range(2, 5)], axis=-1)
        proj_eigen = {Spin.up: proj_eigen[1]}
    else:
        proj_eigen = {Spin.up if k == 1 else Spin.down: v for k, v in proj_eigen.items()}
        proj_mag = None
    elem.clear()
    return (proj_eigen, proj_mag)