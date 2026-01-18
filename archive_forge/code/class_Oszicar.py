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
class Oszicar:
    """
    A basic parser for an OSZICAR output from VASP. In general, while the
    OSZICAR is useful for a quick look at the output from a VASP run, we
    recommend that you use the Vasprun parser instead, which gives far richer
    information about a run.

    Attributes:
        electronic_steps (list): All electronic steps as a list of list of dict. e.g.,
            [[{"rms": 160.0, "E": 4507.24605593, "dE": 4507.2, "N": 1, "deps": -17777.0, "ncg": 16576}, ...], [....]
            where electronic_steps[index] refers the list of electronic steps in one ionic_step,
            electronic_steps[index][subindex] refers to a particular electronic step at subindex in ionic step at
            index. The dict of properties depends on the type of VASP run, but in general, "E", "dE" and "rms" should
            be present in almost all runs.
        ionic_steps (list): All ionic_steps as a list of dict, e.g.,
            [{"dE": -526.36, "E0": -526.36024, "mag": 0.0, "F": -526.36024}, ...]
            This is the typical output from VASP at the end of each ionic step. The stored dict might be different
            depending on the type of VASP run.
    """

    def __init__(self, filename):
        """
        Args:
            filename (str): Filename of file to parse.
        """
        electronic_steps = []
        ionic_steps = []
        ionic_general_pattern = re.compile('(\\w+)=\\s*(\\S+)')
        electronic_pattern = re.compile('\\s*\\w+\\s*:(.*)')

        def smart_convert(header, num):
            try:
                if header in ('N', 'ncg'):
                    return int(num)
                return float(num)
            except ValueError:
                return '--'
        header = []
        with zopen(filename, mode='rt') as fid:
            for line in fid:
                m = electronic_pattern.match(line.strip())
                if m:
                    tokens = m.group(1).split()
                    data = {header[i]: smart_convert(header[i], tokens[i]) for i in range(len(tokens))}
                    if tokens[0] == '1':
                        electronic_steps.append([data])
                    else:
                        electronic_steps[-1].append(data)
                elif re.match('^\\s*N\\s+E\\s*', line.strip()):
                    header = line.strip().replace('d eps', 'deps').split()
                elif line.strip() != '':
                    matches = re.findall(ionic_general_pattern, re.sub('d E ', 'dE', line))
                    ionic_steps.append({key: float(value) for key, value in matches})
        self.electronic_steps = electronic_steps
        self.ionic_steps = ionic_steps

    @property
    def all_energies(self):
        """
        Compilation of all energies from all electronic steps and ionic steps
        as a tuple of list of energies, e.g.,
        ((4507.24605593, 143.824705755, -512.073149912, ...), ...).
        """
        all_energies = []
        for i in range(len(self.electronic_steps)):
            energies = [step['E'] for step in self.electronic_steps[i]]
            energies.append(self.ionic_steps[i]['F'])
            all_energies.append(tuple(energies))
        return tuple(all_energies)

    @property
    @unitized('eV')
    def final_energy(self):
        """Final energy from run."""
        return self.ionic_steps[-1]['E0']

    def as_dict(self):
        """MSONable dict"""
        return {'electronic_steps': self.electronic_steps, 'ionic_steps': self.ionic_steps}