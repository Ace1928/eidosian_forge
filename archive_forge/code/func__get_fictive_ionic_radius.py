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
def _get_fictive_ionic_radius(site: Site, neighbor: PeriodicNeighbor) -> float:
    """
    Get fictive ionic radius.

    Follows equation 1 of:

    Hoppe, Rudolf. "Effective coordination numbers (ECoN) and mean fictive ionic
    radii (MEFIR)." Zeitschrift für Kristallographie-Crystalline Materials
    150.1-4 (1979): 23-52.

    Args:
        site: The central site.
        neighbor neighboring site.

    Returns:
        Hoppe's fictive ionic radius.
    """
    r_h = _get_radius(site)
    if r_h == 0:
        r_h = _get_default_radius(site)
    r_i = _get_radius(neighbor)
    if r_i == 0:
        r_i = _get_default_radius(neighbor)
    return neighbor.nn_distance * (r_h / (r_h + r_i))