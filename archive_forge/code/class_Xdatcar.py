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
class Xdatcar:
    """
    Class representing an XDATCAR file. Only tested with VASP 5.x files.

    Attributes:
        structures (list): List of structures parsed from XDATCAR.
        comment (str): Optional comment string.

    Authors: Ram Balachandran
    """

    def __init__(self, filename, ionicstep_start=1, ionicstep_end=None, comment=None):
        """
        Init a Xdatcar.

        Args:
            filename (str): Filename of input XDATCAR file.
            ionicstep_start (int): Starting number of ionic step.
            ionicstep_end (int): Ending number of ionic step.
            comment (str): Optional comment attached to this set of structures.
        """
        preamble = None
        coords_str = []
        structures = []
        preamble_done = False
        if ionicstep_start < 1:
            raise ValueError('Start ionic step cannot be less than 1')
        if ionicstep_end is not None and ionicstep_start < 1:
            raise ValueError('End ionic step cannot be less than 1')
        ionicstep_cnt = 1
        with zopen(filename, mode='rt') as file:
            for line in file:
                line = line.strip()
                if preamble is None:
                    preamble = [line]
                    title = line
                elif title == line:
                    preamble_done = False
                    poscar = Poscar.from_str('\n'.join([*preamble, 'Direct', *coords_str]))
                    if ionicstep_end is None:
                        if ionicstep_cnt >= ionicstep_start:
                            structures.append(poscar.structure)
                    else:
                        if ionicstep_start <= ionicstep_cnt < ionicstep_end:
                            structures.append(poscar.structure)
                        if ionicstep_cnt >= ionicstep_end:
                            break
                    ionicstep_cnt += 1
                    coords_str = []
                    preamble = [line]
                elif not preamble_done:
                    if line == '' or 'Direct configuration=' in line:
                        preamble_done = True
                        tmp_preamble = [preamble[0]]
                        for i in range(1, len(preamble)):
                            if preamble[0] != preamble[i]:
                                tmp_preamble.append(preamble[i])
                            else:
                                break
                        preamble = tmp_preamble
                    else:
                        preamble.append(line)
                elif line == '' or 'Direct configuration=' in line:
                    poscar = Poscar.from_str('\n'.join([*preamble, 'Direct', *coords_str]))
                    if ionicstep_end is None:
                        if ionicstep_cnt >= ionicstep_start:
                            structures.append(poscar.structure)
                    else:
                        if ionicstep_start <= ionicstep_cnt < ionicstep_end:
                            structures.append(poscar.structure)
                        if ionicstep_cnt >= ionicstep_end:
                            break
                    ionicstep_cnt += 1
                    coords_str = []
                else:
                    coords_str.append(line)
            poscar = Poscar.from_str('\n'.join([*preamble, 'Direct', *coords_str]))
            if ionicstep_end is None:
                if ionicstep_cnt >= ionicstep_start:
                    structures.append(poscar.structure)
            elif ionicstep_start <= ionicstep_cnt < ionicstep_end:
                structures.append(poscar.structure)
        self.structures = structures
        self.comment = comment or self.structures[0].formula

    @property
    def site_symbols(self):
        """
        Sequence of symbols associated with the Xdatcar. Similar to 6th line in
        vasp 5+ Xdatcar.
        """
        syms = [site.specie.symbol for site in self.structures[0]]
        return [a[0] for a in itertools.groupby(syms)]

    @property
    def natoms(self):
        """
        Sequence of number of sites of each type associated with the Poscar.
        Similar to 7th line in vasp 5+ Xdatcar.
        """
        syms = [site.specie.symbol for site in self.structures[0]]
        return [len(tuple(a[1])) for a in itertools.groupby(syms)]

    def concatenate(self, filename, ionicstep_start=1, ionicstep_end=None):
        """
        Concatenate structures in file to Xdatcar.

        Args:
            filename (str): Filename of XDATCAR file to be concatenated.
            ionicstep_start (int): Starting number of ionic step.
            ionicstep_end (int): Ending number of ionic step.
        TODO (rambalachandran): Requires a check to ensure if the new concatenating file
            has the same lattice structure and atoms as the Xdatcar class.
        """
        preamble = None
        coords_str = []
        structures = self.structures
        preamble_done = False
        if ionicstep_start < 1:
            raise ValueError('Start ionic step cannot be less than 1')
        if ionicstep_end is not None and ionicstep_start < 1:
            raise ValueError('End ionic step cannot be less than 1')
        ionicstep_cnt = 1
        with zopen(filename, mode='rt') as file:
            for line in file:
                line = line.strip()
                if preamble is None:
                    preamble = [line]
                elif not preamble_done:
                    if line == '' or 'Direct configuration=' in line:
                        preamble_done = True
                        tmp_preamble = [preamble[0]]
                        for i in range(1, len(preamble)):
                            if preamble[0] != preamble[i]:
                                tmp_preamble.append(preamble[i])
                            else:
                                break
                        preamble = tmp_preamble
                    else:
                        preamble.append(line)
                elif line == '' or 'Direct configuration=' in line:
                    poscar = Poscar.from_str('\n'.join([*preamble, 'Direct', *coords_str]))
                    if ionicstep_end is None:
                        if ionicstep_cnt >= ionicstep_start:
                            structures.append(poscar.structure)
                    elif ionicstep_start <= ionicstep_cnt < ionicstep_end:
                        structures.append(poscar.structure)
                    ionicstep_cnt += 1
                    coords_str = []
                else:
                    coords_str.append(line)
            poscar = Poscar.from_str('\n'.join([*preamble, 'Direct', *coords_str]))
            if ionicstep_end is None:
                if ionicstep_cnt >= ionicstep_start:
                    structures.append(poscar.structure)
            elif ionicstep_start <= ionicstep_cnt < ionicstep_end:
                structures.append(poscar.structure)
        self.structures = structures

    def get_str(self, ionicstep_start: int=1, ionicstep_end: int | None=None, significant_figures: int=8) -> str:
        """
        Write  Xdatcar class to a string.

        Args:
            ionicstep_start (int): Starting number of ionic step.
            ionicstep_end (int): Ending number of ionic step.
            significant_figures (int): Number of significant figures.
        """
        if ionicstep_start < 1:
            raise ValueError('Start ionic step cannot be less than 1')
        if ionicstep_end is not None and ionicstep_end < 1:
            raise ValueError('End ionic step cannot be less than 1')
        lattice = self.structures[0].lattice
        if np.linalg.det(lattice.matrix) < 0:
            lattice = Lattice(-lattice.matrix)
        lines = [self.comment, '1.0', str(lattice)]
        lines.extend((' '.join(self.site_symbols), ' '.join((str(x) for x in self.natoms))))
        format_str = f'{{:.{significant_figures}f}}'
        ionicstep_cnt = 1
        output_cnt = 1
        for cnt, structure in enumerate(self.structures, start=1):
            ionicstep_cnt = cnt
            if ionicstep_end is None:
                if ionicstep_cnt >= ionicstep_start:
                    lines.append(f'Direct configuration={' ' * (7 - len(str(output_cnt)))}{output_cnt}')
                    for site in structure:
                        coords = site.frac_coords
                        line = ' '.join((format_str.format(c) for c in coords))
                        lines.append(line)
                    output_cnt += 1
            elif ionicstep_start <= ionicstep_cnt < ionicstep_end:
                lines.append(f'Direct configuration={' ' * (7 - len(str(output_cnt)))}{output_cnt}')
                for site in structure:
                    coords = site.frac_coords
                    line = ' '.join((format_str.format(c) for c in coords))
                    lines.append(line)
                output_cnt += 1
        return '\n'.join(lines) + '\n'

    def write_file(self, filename, **kwargs):
        """
        Write Xdatcar class into a file.

        Args:
            filename (str): Filename of output XDATCAR file.
            **kwargs: Supported kwargs are the same as those for the
                Xdatcar.get_str method and are passed through directly.
        """
        with zopen(filename, mode='wt') as file:
            file.write(self.get_str(**kwargs))

    def __str__(self):
        return self.get_str()