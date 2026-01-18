from __future__ import annotations
import abc
import itertools
from math import ceil, cos, e, pi, sin, tan
from typing import TYPE_CHECKING, Any
from warnings import warn
import networkx as nx
import numpy as np
import spglib
from monty.dev import requires
from scipy.linalg import sqrtm
from pymatgen.core.lattice import Lattice
from pymatgen.core.operations import MagSymmOp, SymmOp
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer, cite_conventional_cell_algo
class KPathBase(abc.ABC):
    """This is the base class for classes used to generate high-symmetry
    paths in reciprocal space (k-paths) for band structure calculations.
    """

    @abc.abstractmethod
    def __init__(self, structure: Structure, symprec: float=0.01, angle_tolerance=5, atol=1e-05, *args, **kwargs):
        """
        Args:
            structure (Structure): Structure object.
            symprec (float): Tolerance for symmetry finding.
            angle_tolerance (float): Angle tolerance for symmetry finding.
            atol (float): Absolute tolerance used to compare structures
                and determine symmetric equivalence of points and lines in the BZ.
            *args: Other arguments supported by subclasses.
            **kwargs: Other keyword arguments supported by subclasses.
        """
        self._structure = structure
        self._latt = self._structure.lattice
        self._rec_lattice = self._structure.lattice.reciprocal_lattice
        self._kpath: dict[str, Any] | None = None
        self._symprec = symprec
        self._atol = atol
        self._angle_tolerance = angle_tolerance

    @property
    def structure(self):
        """
        Returns:
            The input structure.
        """
        return self._structure

    @property
    def lattice(self):
        """
        Returns:
            The real space lattice.
        """
        return self._latt

    @property
    def rec_lattice(self):
        """
        Returns:
            The reciprocal space lattice.
        """
        return self._rec_lattice

    @property
    def kpath(self):
        """
        Returns:
            The symmetry line path in reciprocal space.
        """
        return self._kpath

    def get_kpoints(self, line_density=20, coords_are_cartesian=True):
        """
        Returns:
            kpoints along the path in Cartesian coordinates
        together with the critical-point labels.
        """
        list_k_points = []
        sym_point_labels = []
        for b in self.kpath['path']:
            for i in range(1, len(b)):
                start = np.array(self.kpath['kpoints'][b[i - 1]])
                end = np.array(self.kpath['kpoints'][b[i]])
                distance = np.linalg.norm(self._rec_lattice.get_cartesian_coords(start) - self._rec_lattice.get_cartesian_coords(end))
                nb = int(ceil(distance * line_density))
                if nb == 0:
                    continue
                sym_point_labels.extend([b[i - 1]] + [''] * (nb - 1) + [b[i]])
                list_k_points += [self._rec_lattice.get_cartesian_coords(start) + float(i) / float(nb) * (self._rec_lattice.get_cartesian_coords(end) - self._rec_lattice.get_cartesian_coords(start)) for i in range(nb + 1)]
        if coords_are_cartesian:
            return (list_k_points, sym_point_labels)
        frac_k_points = [self._rec_lattice.get_fractional_coords(k) for k in list_k_points]
        return (frac_k_points, sym_point_labels)