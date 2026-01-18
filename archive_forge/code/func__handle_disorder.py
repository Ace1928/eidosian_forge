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
def _handle_disorder(structure: Structure, on_disorder: on_disorder_options):
    """What to do in bonding and coordination number analysis if a site is disordered."""
    if all((site.is_ordered for site in structure)):
        return structure
    if on_disorder == 'error':
        raise ValueError(f"Generating StructureGraphs for disordered Structures is unsupported. Pass on_disorder='take majority' | 'take_max_species' | 'error'. 'take_majority_strict' considers only the majority species from each site in the bonding algorithm and raises ValueError in case there is no majority (e.g. as in {{Fe: 0.4, O: 0.4, C: 0.2}}) whereas 'take_majority_drop' just ignores the site altogether when computing bonds as if it didn't exist. 'take_max_species' extracts the first max species on each site (Fe in prev. example since Fe and O have equal occupancy and Fe comes first). 'error' raises an error in case of disordered structure. Offending structure = {structure!r}")
    if on_disorder.startswith('take_'):
        structure = structure.copy()
        for idx, site in enumerate(structure):
            max_specie = max(site.species, key=site.species.get)
            max_val = site.species[max_specie]
            if max_val <= 0.5:
                if on_disorder == 'take_majority_strict':
                    raise ValueError(f'Site {idx} has no majority species, the max species is {max_specie} with occupancy {max_val}')
                if on_disorder == 'take_majority_drop':
                    continue
            site.species = max_specie
    else:
        raise ValueError(f'Unexpected on_disorder = {on_disorder!r}, should be one of {get_args(on_disorder_options)}')
    return structure