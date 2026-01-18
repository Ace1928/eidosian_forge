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
def _get_radius(site):
    """
    An internal method to get the expected radius for a site with
    oxidation state.

    Args:
        site: (Site)

    Returns:
        Oxidation-state dependent radius: ionic, covalent, or atomic.
        Returns 0 if no oxidation state or appropriate radius is found.
    """
    if hasattr(site.specie, 'oxi_state'):
        el = site.specie.element
        oxi = site.specie.oxi_state
        if oxi == 0:
            return _get_default_radius(site)
        if oxi in el.ionic_radii:
            return el.ionic_radii[oxi]
        if int(math.floor(oxi)) in el.ionic_radii and int(math.ceil(oxi)) in el.ionic_radii:
            oxi_low = el.ionic_radii[int(math.floor(oxi))]
            oxi_high = el.ionic_radii[int(math.ceil(oxi))]
            x = oxi - int(math.floor(oxi))
            return (1 - x) * oxi_low + x * oxi_high
        if oxi > 0 and el.average_cationic_radius > 0:
            return el.average_cationic_radius
        if el.average_anionic_radius > 0 > oxi:
            return el.average_anionic_radius
    else:
        warnings.warn('No oxidation states specified on sites! For better results, set the site oxidation states in the structure.')
    return 0