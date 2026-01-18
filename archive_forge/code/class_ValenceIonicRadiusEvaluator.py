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
class ValenceIonicRadiusEvaluator:
    """
    Computes site valences and ionic radii for a structure using bond valence
    analyzer.
    """

    def __init__(self, structure: Structure) -> None:
        """
        Args:
            structure: pymatgen Structure.
        """
        self._structure = structure.copy()
        self._valences = self._get_valences()
        self._ionic_radii = self._get_ionic_radii()

    @property
    def radii(self):
        """List of ionic radii of elements in the order of sites."""
        elems = [site.species_string for site in self._structure]
        return dict(zip(elems, self._ionic_radii))

    @property
    def valences(self):
        """List of oxidation states of elements in the order of sites."""
        el = [site.species_string for site in self._structure]
        return dict(zip(el, self._valences))

    @property
    def structure(self):
        """Returns oxidation state decorated structure."""
        return self._structure.copy()

    def _get_ionic_radii(self):
        """
        Computes ionic radii of elements for all sites in the structure.
        If valence is zero, atomic radius is used.
        """
        radii = []
        vnn = VoronoiNN()

        def nearest_key(sorted_vals: list[int], skey: int) -> int:
            n = bisect_left(sorted_vals, skey)
            if n == len(sorted_vals):
                return sorted_vals[-1]
            if n == 0:
                return sorted_vals[0]
            before = sorted_vals[n - 1]
            after = sorted_vals[n]
            if after - skey < skey - before:
                return after
            return before
        for idx, site in enumerate(self._structure):
            if isinstance(site.specie, Element):
                radius = site.specie.atomic_radius
                if radius is None:
                    radius = site.specie.atomic_radius_calculated
                if radius is None:
                    raise ValueError(f'cannot assign radius to element {site.specie}')
                radii.append(radius)
                continue
            el = site.specie.symbol
            oxi_state = int(round(site.specie.oxi_state))
            coord_no = int(round(vnn.get_cn(self._structure, idx)))
            try:
                tab_oxi_states = sorted(map(int, _ion_radii[el]))
                oxi_state = nearest_key(tab_oxi_states, oxi_state)
                radius = _ion_radii[el][str(oxi_state)][str(coord_no)]
            except KeyError:
                new_coord_no = coord_no + 1 if vnn.get_cn(self._structure, idx) - coord_no > 0 else coord_no - 1
                try:
                    radius = _ion_radii[el][str(oxi_state)][str(new_coord_no)]
                    coord_no = new_coord_no
                except Exception:
                    tab_coords = sorted(map(int, _ion_radii[el][str(oxi_state)]))
                    new_coord_no = nearest_key(tab_coords, coord_no)
                    idx = 0
                    for val in tab_coords:
                        if val > coord_no:
                            break
                        idx = idx + 1
                    if idx == len(tab_coords):
                        key = str(tab_coords[-1])
                        radius = _ion_radii[el][str(oxi_state)][key]
                    elif idx == 0:
                        key = str(tab_coords[0])
                        radius = _ion_radii[el][str(oxi_state)][key]
                    else:
                        key = str(tab_coords[idx - 1])
                        radius1 = _ion_radii[el][str(oxi_state)][key]
                        key = str(tab_coords[idx])
                        radius2 = _ion_radii[el][str(oxi_state)][key]
                        radius = (radius1 + radius2) / 2
            radii.append(radius)
        return radii

    def _get_valences(self):
        """Computes ionic valences of elements for all sites in the structure."""
        try:
            bv = BVAnalyzer()
            self._structure = bv.get_oxi_state_decorated_structure(self._structure)
            valences = bv.get_valences(self._structure)
        except Exception:
            try:
                bv = BVAnalyzer(symm_tol=0)
                self._structure = bv.get_oxi_state_decorated_structure(self._structure)
                valences = bv.get_valences(self._structure)
            except Exception:
                valences = []
                for site in self._structure:
                    if len(site.specie.common_oxidation_states) > 0:
                        valences.append(site.specie.common_oxidation_states[0])
                    else:
                        valences.append(0)
                if sum(valences):
                    valences = [0] * len(self._structure)
                else:
                    self._structure.add_oxidation_state_by_site(valences)
        return valences