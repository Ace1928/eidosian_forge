from __future__ import annotations
import collections
import copy
import math
import tempfile
from typing import TYPE_CHECKING, NamedTuple
import numpy as np
from monty.dev import deprecated
from pymatgen.analysis.bond_valence import BVAnalyzer
from pymatgen.analysis.chemenv.coordination_environments.coordination_geometry_finder import LocalGeometryFinder
from pymatgen.analysis.chemenv.coordination_environments.structure_environments import LightStructureEnvironments
from pymatgen.analysis.local_env import NearNeighbors
from pymatgen.electronic_structure.cohp import CompleteCohp
from pymatgen.electronic_structure.core import Spin
from pymatgen.electronic_structure.plotter import CohpPlotter
from pymatgen.io.lobster import Charge, Icohplist
from pymatgen.util.due import Doi, due
@staticmethod
def _get_atomnumber(atomstring) -> int:
    """
        Return the number of the atom within the initial POSCAR (e.g., Return 0 for "Na1").

        Args:
            atomstring: string such as "Na1"

        Returns:
            int: indicating the position in the POSCAR
        """
    return int(LobsterNeighbors._split_string(atomstring)[1]) - 1