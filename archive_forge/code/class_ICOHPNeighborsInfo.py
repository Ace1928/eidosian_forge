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
class ICOHPNeighborsInfo(NamedTuple):
    """
    Tuple to represent information on relevant bonds
    Args:
        total_icohp (float): sum of icohp values of neighbors to the selected sites [given by the id in structure]
        list_icohps (list): list of summed icohp values for all identified interactions with neighbors
        n_bonds (int): number of identified bonds to the selected sites
        labels (list[str]): labels (from ICOHPLIST) for all identified bonds
        atoms (list[list[str]]): list of list describing the species present in the identified interactions
            (names from ICOHPLIST), e.g., ["Ag3", "O5"]
        central_isites (list[int]): list of the central isite for each identified interaction.
    """
    total_icohp: float
    list_icohps: list[float]
    n_bonds: int
    labels: list[str]
    atoms: list[list[str]]
    central_isites: list[int] | None