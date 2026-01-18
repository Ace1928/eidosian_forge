from __future__ import annotations
import json
import math
import os
import warnings
from bisect import bisect_left
from collections import defaultdict, namedtuple
from copy import deepcopy
from functools import lru_cache
from math import acos, asin, atan2, cos, exp, fabs, pi, pow, sin, sqrt
from typing import TYPE_CHECKING, Any, Literal, get_args
import numpy as np
from monty.dev import deprecated, requires
from monty.serialization import loadfn
from ruamel.yaml import YAML
from scipy.spatial import Voronoi
from pymatgen.analysis.bond_valence import BV_PARAMS, BVAnalyzer
from pymatgen.analysis.graphs import MoleculeGraph, StructureGraph
from pymatgen.analysis.molecule_structure_comparator import CovalentRadius
from pymatgen.core import Element, IStructure, PeriodicNeighbor, PeriodicSite, Site, Species, Structure
def _get_mean_fictive_ionic_radius(fictive_ionic_radii: list[float], minimum_fir: float | None=None) -> float:
    """
    Returns the mean fictive ionic radius.

    Follows equation 2:

    Hoppe, Rudolf. "Effective coordination numbers (ECoN) and mean fictive ionic
    radii (MEFIR)." Zeitschrift für Kristallographie-Crystalline Materials
    150.1-4 (1979): 23-52.

    Args:
        fictive_ionic_radii: List of fictive ionic radii for a center site
            and its neighbors.
        minimum_fir: Minimum fictive ionic radius to use.

    Returns:
        Hoppe's mean fictive ionic radius.
    """
    if not minimum_fir:
        minimum_fir = min(fictive_ionic_radii)
    weighted_sum = 0.0
    total_sum = 0.0
    for fir in fictive_ionic_radii:
        weighted_sum += fir * exp(1 - (fir / minimum_fir) ** 6)
        total_sum += exp(1 - (fir / minimum_fir) ** 6)
    return weighted_sum / total_sum