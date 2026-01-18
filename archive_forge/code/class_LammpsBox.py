from __future__ import annotations
import itertools
import re
import warnings
from io import StringIO
from pathlib import Path
from typing import TYPE_CHECKING, Literal
import numpy as np
import pandas as pd
from monty.io import zopen
from monty.json import MSONable
from monty.serialization import loadfn
from ruamel.yaml import YAML
from pymatgen.core import Element, Lattice, Molecule, Structure
from pymatgen.core.operations import SymmOp
from pymatgen.util.io_utils import clean_lines
class LammpsBox(MSONable):
    """Object for representing a simulation box in LAMMPS settings."""

    def __init__(self, bounds: Sequence, tilt: Sequence | None=None) -> None:
        """
        Args:
            bounds: A (3, 2) array/list of floats setting the
                boundaries of simulation box.
            tilt: A (3,) array/list of floats setting the tilt of
                simulation box. Default to None, i.e., use an
                orthogonal box.
        """
        bounds_arr = np.array(bounds)
        assert bounds_arr.shape == (3, 2), f'Expecting a (3, 2) array for bounds, got {bounds_arr.shape}'
        self.bounds = bounds_arr.tolist()
        matrix = np.diag(bounds_arr[:, 1] - bounds_arr[:, 0])
        self.tilt = None
        if tilt is not None:
            tilt_arr = np.array(tilt)
            assert tilt_arr.shape == (3,), f'Expecting a (3,) array for box_tilt, got {tilt_arr.shape}'
            self.tilt = tilt_arr.tolist()
            matrix[1, 0] = tilt_arr[0]
            matrix[2, 0] = tilt_arr[1]
            matrix[2, 1] = tilt_arr[2]
        self._matrix = matrix

    def __str__(self) -> str:
        return self.get_str()

    def __repr__(self) -> str:
        return self.get_str()

    @property
    def volume(self) -> float:
        """Volume of simulation box."""
        matrix = self._matrix
        return np.dot(np.cross(matrix[0], matrix[1]), matrix[2])

    def get_str(self, significant_figures: int=6) -> str:
        """
        Returns the string representation of simulation box in LAMMPS
        data file format.

        Args:
            significant_figures (int): No. of significant figures to
                output for box settings. Default to 6.

        Returns:
            String representation
        """
        ph = f'{{:.{significant_figures}f}}'
        lines = []
        for bound, d in zip(self.bounds, 'xyz'):
            fillers = bound + [d] * 2
            bound_format = ' '.join([ph] * 2 + [' {}lo {}hi'])
            lines.append(bound_format.format(*fillers))
        if self.tilt:
            tilt_format = ' '.join([ph] * 3 + [' xy xz yz'])
            lines.append(tilt_format.format(*self.tilt))
        return '\n'.join(lines)

    def get_box_shift(self, i: Sequence[int]) -> np.ndarray:
        """
        Calculates the coordinate shift due to PBC.

        Args:
            i: A (n, 3) integer array containing the labels for box
            images of n entries.

        Returns:
            Coordinate shift array with the same shape of i
        """
        return np.inner(i, self._matrix.T)

    def to_lattice(self) -> Lattice:
        """
        Converts the simulation box to a more powerful Lattice backend.
        Note that Lattice is always periodic in 3D space while a
        simulation box is not necessarily periodic in all dimensions.

        Returns:
            Lattice
        """
        return Lattice(self._matrix)