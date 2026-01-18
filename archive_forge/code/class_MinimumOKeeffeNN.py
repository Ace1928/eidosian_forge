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
class MinimumOKeeffeNN(NearNeighbors):
    """
    Determine near-neighbor sites and coordination number using the
    neighbor(s) at closest relative distance, d_min_OKeffee, plus some
    relative tolerance, where bond valence parameters from O'Keeffe's
    bond valence method (J. Am. Chem. Soc. 1991, 3226-3229) are used
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
        return True

    @property
    def extend_structure_molecules(self) -> bool:
        """
        Boolean property: Do Molecules need to be converted to Structures to use
        this NearNeighbors class? Note: this property is not defined for classes
        for which molecules_allowed is False.
        """
        return True

    def get_nn_info(self, structure: Structure, n: int):
        """
        Get all near-neighbor sites as well as the associated image locations
        and weights of the site with index n using the closest relative
        neighbor distance-based method with O'Keeffe parameters.

        Args:
            structure (Structure): input structure.
            n (int): index of site for which to determine near
                neighbors.

        Returns:
            siw (list of tuples (Site, array, float)): tuples, each one
                of which represents a neighbor site, its image location,
                and its weight.
        """
        site = structure[n]
        neighs_dists = structure.get_neighbors(site, self.cutoff)
        try:
            eln = site.specie.element
        except Exception:
            eln = site.species_string
        reldists_neighs = []
        for nn in neighs_dists:
            neigh = nn
            dist = nn.nn_distance
            try:
                el2 = neigh.specie.element
            except Exception:
                el2 = neigh.species_string
            reldists_neighs.append([dist / get_okeeffe_distance_prediction(eln, el2), neigh])
        siw = []
        min_reldist = min((reldist for reldist, neigh in reldists_neighs))
        for reldist, s in reldists_neighs:
            if reldist < (1 + self.tol) * min_reldist:
                w = min_reldist / reldist
                siw.append({'site': s, 'image': self._get_image(structure, s), 'weight': w, 'site_index': self._get_original_site(structure, s)})
        return siw