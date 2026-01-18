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
class MinimumVIRENN(NearNeighbors):
    """
    Determine near-neighbor sites and coordination number using the
    neighbor(s) at closest relative distance, d_min_VIRE, plus some
    relative tolerance, where atom radii from the
    ValenceIonicRadiusEvaluator (VIRE) are used
    to calculate relative distances.
    """

    def __init__(self, tol: float=0.1, cutoff=10) -> None:
        """
        Args:
            tol (float): tolerance parameter for neighbor identification
                (default: 0.1).
            cutoff (float): cutoff radius in Angstrom to look for trial
                near-neighbor sites (default: 10).
        """
        self.tol = tol
        self.cutoff = cutoff

    @property
    def structures_allowed(self) -> bool:
        """
        Boolean property: can this NearNeighbors class be used with Structure
        objects?
        """
        return True

    @property
    def molecules_allowed(self) -> bool:
        """
        Boolean property: can this NearNeighbors class be used with Molecule
        objects?
        """
        return False

    def get_nn_info(self, structure: Structure, n: int):
        """
        Get all near-neighbor sites as well as the associated image locations
        and weights of the site with index n using the closest relative
        neighbor distance-based method with VIRE atomic/ionic radii.

        Args:
            structure (Structure): input structure.
            n (int): index of site for which to determine near
                neighbors.

        Returns:
            siw (list of tuples (Site, array, float)): tuples, each one
                of which represents a neighbor site, its image location,
                and its weight.
        """
        vire = _get_vire(structure)
        site = vire.structure[n]
        neighs_dists = vire.structure.get_neighbors(site, self.cutoff)
        rn = vire.radii[vire.structure[n].species_string]
        reldists_neighs = []
        for nn in neighs_dists:
            reldists_neighs.append([nn.nn_distance / (vire.radii[nn.species_string] + rn), nn])
        siw = []
        min_reldist = min((reldist for reldist, neigh in reldists_neighs))
        for reldist, s in reldists_neighs:
            if reldist < (1 + self.tol) * min_reldist:
                w = min_reldist / reldist
                siw.append({'site': s, 'image': self._get_image(vire.structure, s), 'weight': w, 'site_index': self._get_original_site(vire.structure, s)})
        return siw