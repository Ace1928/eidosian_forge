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
def _get_nn_shell_info(self, structure, all_nn_info, site_idx, shell, _previous_steps=frozenset(), _cur_image=(0, 0, 0)):
    """Private method for computing the neighbor shell information.

        Args:
            structure (Structure) - Structure being assessed
            all_nn_info ([[dict]]) - Results from `get_all_nn_info`
            site_idx (int) - index of site for which to determine neighbor
                information.
            shell (int) - Which neighbor shell to retrieve (1 == 1st NN shell)
            _previous_steps ({(site_idx, image}) - Internal use only: Set of
                sites that have already been traversed.
            _cur_image (tuple) - Internal use only Image coordinates of current atom

        Returns:
            list of dictionaries. Each entry in the list is information about
                a certain neighbor in the structure, in the same format as
                `get_nn_info`. Does not update the site positions
        """
    if shell <= 0:
        raise ValueError('Shell must be positive')
    _previous_steps = _previous_steps | {(site_idx, _cur_image)}
    possible_steps = list(all_nn_info[site_idx])
    for idx, step in enumerate(possible_steps):
        step = dict(step)
        step['image'] = tuple(np.add(step['image'], _cur_image).tolist())
        possible_steps[idx] = step
    allowed_steps = [x for x in possible_steps if (x['site_index'], x['image']) not in _previous_steps]
    if shell == 1:
        return allowed_steps
    terminal_neighbors = [self._get_nn_shell_info(structure, all_nn_info, x['site_index'], shell - 1, _previous_steps, x['image']) for x in allowed_steps]
    all_sites = {}
    for first_site, term_sites in zip(allowed_steps, terminal_neighbors):
        for term_site in term_sites:
            key = (term_site['site_index'], tuple(term_site['image']))
            term_site['weight'] *= first_site['weight']
            value = all_sites.get(key)
            if value is not None:
                value['weight'] += term_site['weight']
            else:
                value = term_site
            all_sites[key] = value
    return list(all_sites.values())