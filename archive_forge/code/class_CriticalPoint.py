from __future__ import annotations
import logging
import os
import subprocess
import warnings
from enum import Enum, unique
from glob import glob
from shutil import which
from typing import TYPE_CHECKING
import numpy as np
from monty.dev import requires
from monty.json import MSONable
from monty.serialization import loadfn
from monty.tempfile import ScratchDir
from scipy.spatial import KDTree
from pymatgen.analysis.graphs import StructureGraph
from pymatgen.core import DummySpecies
from pymatgen.io.vasp.inputs import Potcar
from pymatgen.io.vasp.outputs import Chgcar, VolumetricData
from pymatgen.util.due import Doi, due
class CriticalPoint(MSONable):
    """Access information about a critical point and the field values at that point."""

    def __init__(self, index, type, frac_coords, point_group, multiplicity, field, field_gradient, coords=None, field_hessian=None):
        """Class to characterize a critical point from a topological
        analysis of electron charge density.

        Note this class is usually associated with a Structure, so
        has information on multiplicity/point group symmetry.

        Args:
            index: index of point
            type: type of point, given as a string
            coords: Cartesian coordinates in Angstroms
            frac_coords: fractional coordinates
            point_group: point group associated with critical point
            multiplicity: number of equivalent critical points
            field: value of field at point (f)
            field_gradient: gradient of field at point (grad f)
            field_hessian: hessian of field at point (del^2 f)
        """
        self.index = index
        self._type = type
        self.coords = coords
        self.frac_coords = frac_coords
        self.point_group = point_group
        self.multiplicity = multiplicity
        self.field = field
        self.field_gradient = field_gradient
        self.field_hessian = field_hessian

    @property
    def type(self):
        """Returns: Instance of CriticalPointType."""
        return CriticalPointType(self._type)

    def __str__(self):
        return f'Critical Point: {self.type.name} ({self.frac_coords})'

    @property
    def laplacian(self) -> float:
        """Returns: The Laplacian of the field at the critical point."""
        return np.trace(self.field_hessian)

    @property
    def ellipticity(self):
        """Most meaningful for bond critical points, can be physically interpreted as e.g.
        degree of pi-bonding in organic molecules. Consult literature for more info.

        Returns:
            float: The ellipticity of the field at the critical point.
        """
        eig, _ = np.linalg.eig(self.field_hessian)
        eig.sort()
        return eig[0] / eig[1] - 1