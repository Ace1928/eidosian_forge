from __future__ import annotations
import logging
import os
import re
import warnings
from glob import glob
from itertools import chain
import numpy as np
import pandas as pd
from monty.io import zopen
from monty.json import MSONable, jsanitize
from monty.re import regrep
from pymatgen.core.structure import Molecule, Structure
from pymatgen.core.units import Ha_to_eV
from pymatgen.electronic_structure.bandstructure import BandStructure, BandStructureSymmLine
from pymatgen.electronic_structure.core import Orbital, Spin
from pymatgen.electronic_structure.dos import CompleteDos, Dos
from pymatgen.io.cp2k.inputs import Keyword
from pymatgen.io.cp2k.sets import Cp2kInput
from pymatgen.io.cp2k.utils import natural_keys, postprocessor
from pymatgen.io.xyz import XYZ
@staticmethod
def _gauss_smear(densities, energies, npts, width):
    """Return a gaussian smeared DOS"""
    if not width:
        return densities
    dct = np.zeros(npts)
    e_s = np.linspace(min(energies), max(energies), npts)
    for e, _pd in zip(energies, densities):
        weight = np.exp(-((e_s - e) / width) ** 2) / (np.sqrt(np.pi) * width)
        dct += _pd * weight
    return dct