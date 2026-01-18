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