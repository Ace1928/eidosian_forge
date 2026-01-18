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
class IsayevNN(VoronoiNN):
    """
    Uses the algorithm defined in 10.1038/ncomms15679.

    Sites are considered neighbors if (i) they share a Voronoi facet and (ii) the
    bond distance is less than the sum of the Cordero covalent radii + 0.25 Å.
    """

    def __init__(self, tol: float=0.25, targets: Element | list[Element] | None=None, cutoff: float=13.0, allow_pathological: bool=False, extra_nn_info: bool=True, compute_adj_neighbors: bool=True):
        """
        Args:
            tol: Tolerance in Å for bond distances that are considered coordinated.
            targets: Target element(s).
            cutoff: Cutoff radius in Angstrom to look for near-neighbor atoms.
            allow_pathological: Whether to allow infinite vertices in Voronoi
                coordination.
            extra_nn_info: Add all polyhedron info to `get_nn_info`.
            compute_adj_neighbors: Whether to compute which neighbors are adjacent. Turn
                off for faster performance.
        """
        super().__init__()
        self.tol = tol
        self.cutoff = cutoff
        self.allow_pathological = allow_pathological
        self.targets = targets
        self.extra_nn_info = extra_nn_info
        self.compute_adj_neighbors = compute_adj_neighbors

    def get_nn_info(self, structure: Structure, n: int) -> list[dict[str, Any]]:
        """
        Get all near-neighbor site information.

        Gets the associated image locations and weights of the site with index n
        in structure using Voronoi decomposition and distance cutoff.

        Args:
            structure: Input structure.
            n: Index of site for which to determine near-neighbor sites.

        Returns:
            List of dicts containing the near-neighbor information. Each dict has the
            keys:

            - "site": The near-neighbor site.
            - "image": The periodic image of the near-neighbor site.
            - "weight": The face weight of the Voronoi decomposition.
            - "site_index": The index of the near-neighbor site in the original
              structure.
        """
        nns = self.get_voronoi_polyhedra(structure, n)
        return self._filter_nns(structure, n, nns)

    def get_all_nn_info(self, structure: Structure) -> list[list[dict[str, Any]]]:
        """
        Args:
            structure (Structure): input structure.

        Returns:
            List of near neighbor information for each site. See get_nn_info for the
            format of the data for each site.
        """
        all_nns = self.get_all_voronoi_polyhedra(structure)
        return [self._filter_nns(structure, n, nns) for n, nns in enumerate(all_nns)]

    def _filter_nns(self, structure: Structure, n: int, nns: dict[str, Any]) -> list[dict[str, Any]]:
        """Extract and filter the NN info into the format needed by NearestNeighbors.

        Args:
            structure: The structure.
            n: The central site index.
            nns: Nearest neighbor information for the structure.

        Returns:
            See get_nn_info for the format of the returned data.
        """
        targets = structure.elements if self.targets is None else self.targets
        site = structure[n]
        siw = []
        max_weight = max((nn['area'] for nn in nns.values()))
        for nstats in nns.values():
            nn = nstats.pop('site')
            cov_distance = _get_default_radius(site) + _get_default_radius(nn)
            nn_distance = np.linalg.norm(site.coords - nn.coords)
            if _is_in_targets(nn, targets) and nn_distance <= cov_distance + self.tol:
                nn_info = {'site': nn, 'image': self._get_image(structure, nn), 'weight': nstats['area'] / max_weight, 'site_index': self._get_original_site(structure, nn)}
                if self.extra_nn_info:
                    nn_info['poly_info'] = nstats
                siw.append(nn_info)
        return siw