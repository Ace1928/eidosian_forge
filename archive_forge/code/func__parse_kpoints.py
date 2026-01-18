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
def _parse_kpoints(elem):
    e = elem
    if elem.find('generation'):
        e = elem.find('generation')
    k = Kpoints('Kpoints from vasprun.xml')
    k.style = Kpoints.supported_modes.from_str(e.attrib.get('param', 'Reciprocal'))
    for v in e.findall('v'):
        name = v.attrib.get('name')
        tokens = v.text.split()
        if name == 'divisions':
            k.kpts = [[int(i) for i in tokens]]
        elif name == 'usershift':
            k.kpts_shift = [float(i) for i in tokens]
        elif name in {'genvec1', 'genvec2', 'genvec3', 'shift'}:
            setattr(k, name, [float(i) for i in tokens])
    for va in elem.findall('varray'):
        name = va.attrib['name']
        if name == 'kpointlist':
            actual_kpoints = _parse_vasp_array(va)
        elif name == 'weights':
            weights = [i[0] for i in _parse_vasp_array(va)]
    elem.clear()
    if k.style == Kpoints.supported_modes.Reciprocal:
        k = Kpoints(comment='Kpoints from vasprun.xml', style=Kpoints.supported_modes.Reciprocal, num_kpts=len(k.kpts), kpts=actual_kpoints, kpts_weights=weights)
    return (k, actual_kpoints, weights)