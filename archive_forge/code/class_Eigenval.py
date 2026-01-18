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
class Eigenval:
    """
    Object for reading EIGENVAL file.

    Attributes:
        filename (str): String containing input filename.
        occu_tol (float): Tolerance for determining occupation in band properties.
        ispin (int): Spin polarization tag.
        nelect (int): Number of electrons.
        nkpt (int): Number of kpoints.
        nbands (int): Number of bands.
        kpoints (list): List of kpoints.
        kpoints_weights (list): Weights of each kpoint in the BZ, should sum to 1.
        eigenvalues (dict): Eigenvalues as a dict of {(spin): np.ndarray(shape=(nkpt, nbands, 2))}.
            This representation is based on actual ordering in VASP and is meant as an intermediate representation
            to be converted into proper objects. The kpoint index is 0-based (unlike the 1-based indexing in VASP).
    """

    def __init__(self, filename, occu_tol=1e-08, separate_spins=False):
        """
        Reads input from filename to construct Eigenval object.

        Args:
            filename (str):     filename of EIGENVAL to read in
            occu_tol (float):   tolerance for determining band gap
            separate_spins (bool):   whether the band gap, CBM, and VBM should be
                reported for each individual spin channel. Defaults to False,
                which computes the eigenvalue band properties independent of
                the spin orientation. If True, the calculation must be spin-polarized.
        """
        self.filename = filename
        self.occu_tol = occu_tol
        self.separate_spins = separate_spins
        with zopen(filename, mode='r') as file:
            self.ispin = int(file.readline().split()[-1])
            for _ in range(4):
                file.readline()
            self.nelect, self.nkpt, self.nbands = list(map(int, file.readline().split()))
            self.kpoints = []
            self.kpoints_weights = []
            if self.ispin == 2:
                self.eigenvalues = {Spin.up: np.zeros((self.nkpt, self.nbands, 2)), Spin.down: np.zeros((self.nkpt, self.nbands, 2))}
            else:
                self.eigenvalues = {Spin.up: np.zeros((self.nkpt, self.nbands, 2))}
            ikpt = -1
            for line in file:
                if re.search('(\\s+[\\-+0-9eE.]+){4}', str(line)):
                    ikpt += 1
                    kpt = list(map(float, line.split()))
                    self.kpoints.append(kpt[:-1])
                    self.kpoints_weights.append(kpt[-1])
                    for i in range(self.nbands):
                        sl = list(map(float, file.readline().split()))
                        if len(sl) == 3:
                            self.eigenvalues[Spin.up][ikpt, i, 0] = sl[1]
                            self.eigenvalues[Spin.up][ikpt, i, 1] = sl[2]
                        elif len(sl) == 5:
                            self.eigenvalues[Spin.up][ikpt, i, 0] = sl[1]
                            self.eigenvalues[Spin.up][ikpt, i, 1] = sl[3]
                            self.eigenvalues[Spin.down][ikpt, i, 0] = sl[2]
                            self.eigenvalues[Spin.down][ikpt, i, 1] = sl[4]

    @property
    def eigenvalue_band_properties(self):
        """
        Band properties from the eigenvalues as a tuple,
        (band gap, cbm, vbm, is_band_gap_direct). In the case of separate_spins=True,
        the band gap, cbm, vbm, and is_band_gap_direct are each lists of length 2,
        with index 0 representing the spin-up channel and index 1 representing
        the spin-down channel.
        """
        vbm = -float('inf')
        vbm_kpoint = None
        cbm = float('inf')
        cbm_kpoint = None
        vbm_spins = []
        vbm_spins_kpoints = []
        cbm_spins = []
        cbm_spins_kpoints = []
        if self.separate_spins and len(self.eigenvalues) != 2:
            raise ValueError('The separate_spins flag can only be True if ISPIN = 2')
        for d in self.eigenvalues.values():
            if self.separate_spins:
                vbm = -float('inf')
                cbm = float('inf')
            for k, val in enumerate(d):
                for eigenval, occu in val:
                    if occu > self.occu_tol and eigenval > vbm:
                        vbm = eigenval
                        vbm_kpoint = k
                    elif occu <= self.occu_tol and eigenval < cbm:
                        cbm = eigenval
                        cbm_kpoint = k
            if self.separate_spins:
                vbm_spins.append(vbm)
                vbm_spins_kpoints.append(vbm_kpoint)
                cbm_spins.append(cbm)
                cbm_spins_kpoints.append(cbm_kpoint)
        if self.separate_spins:
            return ([max(cbm_spins[0] - vbm_spins[0], 0), max(cbm_spins[1] - vbm_spins[1], 0)], [cbm_spins[0], cbm_spins[1]], [vbm_spins[0], vbm_spins[1]], [vbm_spins_kpoints[0] == cbm_spins_kpoints[0], vbm_spins_kpoints[1] == cbm_spins_kpoints[1]])
        return (max(cbm - vbm, 0), cbm, vbm, vbm_kpoint == cbm_kpoint)