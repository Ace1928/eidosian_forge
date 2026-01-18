from __future__ import annotations
import collections
import contextlib
import functools
import inspect
import io
import itertools
import json
import math
import os
import random
import re
import sys
import warnings
from abc import ABC, abstractmethod
from fnmatch import fnmatch
from inspect import isclass
from io import StringIO
from typing import TYPE_CHECKING, Any, Callable, Literal, SupportsIndex, cast, get_args
import numpy as np
from monty.dev import deprecated
from monty.io import zopen
from monty.json import MSONable
from numpy import cross, eye
from numpy.linalg import norm
from ruamel.yaml import YAML
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.linalg import expm, polar
from scipy.spatial.distance import squareform
from tabulate import tabulate
from pymatgen.core.bonds import CovalentBond, get_bond_length
from pymatgen.core.composition import Composition
from pymatgen.core.lattice import Lattice, get_points_in_spheres
from pymatgen.core.operations import SymmOp
from pymatgen.core.periodic_table import DummySpecies, Element, Species, get_el_sp
from pymatgen.core.sites import PeriodicSite, Site
from pymatgen.core.units import Length, Mass
from pymatgen.electronic_structure.core import Magmom
from pymatgen.symmetry.maggroups import MagneticSpaceGroup
from pymatgen.util.coord import all_distances, get_angle, lattice_points_in_supercell
class PeriodicNeighbor(PeriodicSite):
    """Simple PeriodicSite subclass to contain a neighboring atom that skips all
    the unnecessary checks for speed. Can be used as a fixed-length tuple of
    size 4 to retain backwards compatibility with past use cases.

        (site, distance, index, image).

    In future, usage should be to call attributes, e.g., PeriodicNeighbor.index,
    PeriodicNeighbor.distance, etc.
    """

    def __init__(self, species: Composition, coords: np.ndarray, lattice: Lattice, properties: dict | None=None, nn_distance: float=0.0, index: int=0, image: tuple=(0, 0, 0), label: str | None=None) -> None:
        """
        Args:
            species (Composition): Same as PeriodicSite
            coords (np.ndarray): Same as PeriodicSite, but must be fractional.
            lattice (Lattice): Same as PeriodicSite
            properties (dict, optional): Same as PeriodicSite. Defaults to None.
            nn_distance (float, optional): Distance to some other Site.. Defaults to 0.0.
            index (int, optional): Index within structure.. Defaults to 0.
            image (tuple, optional): PeriodicImage. Defaults to (0, 0, 0).
            label (str, optional): Label for the site. Defaults to None.
        """
        self._lattice = lattice
        self._frac_coords = coords
        self._species = species
        self.properties = properties or {}
        self.nn_distance = nn_distance
        self.index = index
        self.image = image
        self._label = label

    @property
    def coords(self) -> np.ndarray:
        """Cartesian coords."""
        return self._lattice.get_cartesian_coords(self._frac_coords)

    def __len__(self) -> int:
        """Make neighbor Tuple-like to retain backwards compatibility."""
        return 4

    def __getitem__(self, idx: int | slice):
        """Make neighbor Tuple-like to retain backwards compatibility."""
        return (self, self.nn_distance, self.index, self.image)[idx]

    def as_dict(self) -> dict:
        """Note that method calls the super of Site, which is MSONable itself."""
        return super(Site, self).as_dict()

    @classmethod
    def from_dict(cls, dct: dict) -> Self:
        """Returns a PeriodicNeighbor from a dict.

        Args:
            dct: MSONable dict format.

        Returns:
            PeriodicNeighbor
        """
        return super(Site, cls).from_dict(dct)