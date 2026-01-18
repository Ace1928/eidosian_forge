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
@dataclass
class WSWQ(MSONable):
    """
    Class for reading a WSWQ file.
    The WSWQ file is used to calculation the wave function overlaps between
        - W: Wavefunctions in the current directory's WAVECAR file
        - WQ: Wavefunctions stored in a filed named the WAVECAR.qqq.

    The overlap is computed using the overlap operator S
    which make the PAW wavefunctions orthogonormal:
        <W_k,m| S | W_k,n> = \\delta_{mn}

    The WSWQ file contains matrix elements of the overlap operator S evaluated
    between the planewave wavefunctions W and WQ:
        COVL_k,mn = < W_s,k,m | S | WQ_s,k,n >

    The indices of WSWQ.data are:
        [spin][kpoint][band_i][band_j]

    Attributes:
        nspin: Number of spin channels
        nkpoints: Number of k-points
        nbands: Number of bands
        me_real: Real part of the overlap matrix elements
        me_imag: Imaginary part of the overlap matrix elements
    """
    nspin: int
    nkpoints: int
    nbands: int
    me_real: np.ndarray
    me_imag: np.ndarray

    @property
    def data(self):
        """Complex overlap matrix."""
        return self.me_real + 1j * self.me_imag

    @classmethod
    def from_file(cls, filename: str) -> Self:
        """Constructs a WSWQ object from a file.

        Args:
            filename (str): Name of WSWQ file.

        Returns:
            WSWQ object.
        """
        spin_res = regrep(filename, {'spin': 'spin\\s*=\\s*(\\d+)\\s?\\,\\s?kpoint\\s*=\\s*(\\d+)'}, reverse=True, terminate_on_match=True, postprocess=int)['spin']
        (nspin, nkpoints), _ = spin_res[0]
        ij_res = regrep(filename, {'ij': 'i\\s*=\\s*(\\d+)\\s?\\,\\s?j\\s*=\\s*(\\d+)'}, reverse=True, terminate_on_match=True, postprocess=int)['ij']
        (nbands, _), _ = ij_res[0]
        data_res = regrep(filename, {'data': '\\:\\s*([-+]?\\d*\\.\\d+)\\s+([-+]?\\d*\\.\\d+)'}, reverse=False, terminate_on_match=False, postprocess=float)['data']
        assert len(data_res) == nspin * nkpoints * nbands * nbands
        data = np.array([complex(real_part, img_part) for (real_part, img_part), _ in data_res])
        data = data.reshape((nspin, nkpoints, nbands, nbands))
        data = np.swapaxes(data, 2, 3)
        return cls(nspin=nspin, nkpoints=nkpoints, nbands=nbands, me_real=np.real(data), me_imag=np.imag(data))