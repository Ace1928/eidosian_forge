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
class MinimumDistanceNN(NearNeighbors):
    """
    Determine near-neighbor sites and coordination number using the
    nearest neighbor(s) at distance, d_min, plus all neighbors
    within a distance (1 + tol) * d_min, where tol is a
    (relative) distance tolerance parameter.
    """

    def __init__(self, tol: float=0.1, cutoff=10, get_all_sites=False) -> None:
        """
        Args:
            tol (float): tolerance parameter for neighbor identification
                (default: 0.1).
            cutoff (float): cutoff radius in Angstrom to look for trial
                near-neighbor sites (default: 10).
            get_all_sites (bool): If this is set to True then the neighbor
                sites are only determined by the cutoff radius, tol is ignored.
        """
        self.tol = tol
        self.cutoff = cutoff
        self.get_all_sites = get_all_sites

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
        return True

    @property
    def extend_structure_molecules(self) -> bool:
        """
        Boolean property: Do Molecules need to be converted to Structures to use
        this NearNeighbors class? Note: this property is not defined for classes
        for which molecules_allowed is False.
        """
        return True

    def get_nn_info(self, structure: Structure, n: int) -> list[dict[str, Any]]:
        """
        Get all near-neighbor sites as well as the associated image locations
        and weights of the site with index n using the closest neighbor
        distance-based method.

        Args:
            structure (Structure): input structure.
            n (int): index of site for which to determine near
                neighbors.

        Returns:
            siw (list[dict]): dicts with (Site, array, float) each one of which represents a
                neighbor site, its image location, and its weight.
        """
        site = structure[n]
        neighs_dists = structure.get_neighbors(site, self.cutoff)
        is_periodic = isinstance(structure, (Structure, IStructure))
        siw = []
        if self.get_all_sites:
            for nn in neighs_dists:
                weight = nn.nn_distance
                siw.append({'site': nn, 'image': self._get_image(structure, nn) if is_periodic else None, 'weight': weight, 'site_index': self._get_original_site(structure, nn)})
        else:
            min_dist = min((nn.nn_distance for nn in neighs_dists))
            for nn in neighs_dists:
                dist = nn.nn_distance
                if dist < (1 + self.tol) * min_dist:
                    weight = min_dist / dist
                    siw.append({'site': nn, 'image': self._get_image(structure, nn) if is_periodic else None, 'weight': weight, 'site_index': self._get_original_site(structure, nn)})
        return siw