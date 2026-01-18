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
class EconNN(NearNeighbors):
    """
    Determines the average effective coordination number for each cation in a
    given structure using Hoppe's algorithm.

    This method follows the procedure outlined in:

    Hoppe, Rudolf. "Effective coordination numbers (ECoN) and mean fictive ionic
    radii (MEFIR)." Zeitschrift für Kristallographie-Crystalline Materials
    150.1-4 (1979): 23-52.
    """

    def __init__(self, tol: float=0.2, cutoff: float=10.0, cation_anion: bool=False, use_fictive_radius: bool=False):
        """
        Args:
            tol: Tolerance parameter for bond determination.
            cutoff: Cutoff radius in Angstrom to look for near-neighbor atoms.
            cation_anion: If set to True, will restrict bonding targets to
                sites with opposite or zero charge. Requires an oxidation states
                on all sites in the structure.
            use_fictive_radius: Whether to use the fictive radius in the
                EcoN calculation. If False, the bond distance will be used.
        """
        self.tol = tol
        self.cutoff = cutoff
        self.cation_anion = cation_anion
        self.use_fictive_radius = use_fictive_radius

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
        and weights of the site with index n in structure.

        Args:
            structure (Structure): input structure.
            n (int): index of site for which to determine near-neighbor sites.

        Returns:
            siw (list of tuples (Site, array, float)): tuples, each one
                of which represents a coordinated site, its image location,
                and its weight.
        """
        site = structure[n]
        neighbors = structure.get_neighbors(site, self.cutoff)
        oxi_state = getattr(site.specie, 'oxi_state', None)
        if self.cation_anion and oxi_state is not None:
            if oxi_state >= 0:
                neighbors = [n for n in neighbors if n.oxi_state <= 0]
            elif oxi_state < 0:
                neighbors = [nghbr for nghbr in neighbors if nghbr.oxi_state >= 0]
        if self.use_fictive_radius:
            firs = [_get_fictive_ionic_radius(site, neighbor) for neighbor in neighbors]
        else:
            firs = [neighbor.nn_distance for neighbor in neighbors]
        mefir = _get_mean_fictive_ionic_radius(firs)
        prev_mefir = float('inf')
        while abs(prev_mefir - mefir) > 0.0001:
            prev_mefir = mefir
            mefir = _get_mean_fictive_ionic_radius(firs, minimum_fir=mefir)
        siw = []
        for nn, fir in zip(neighbors, firs):
            if nn.nn_distance < self.cutoff:
                w = exp(1 - (fir / mefir) ** 6)
                if w > self.tol:
                    bonded_site = {'site': nn, 'image': self._get_image(structure, nn), 'weight': w, 'site_index': self._get_original_site(structure, nn)}
                    siw.append(bonded_site)
        return siw