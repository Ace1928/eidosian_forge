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
class Critic2NN(NearNeighbors):
    """
    Performs a topological analysis using critic2 to obtain neighbor information, using a
    sum of atomic charge densities. If an actual charge density is available (e.g. from a
    VASP CHGCAR), see Critic2Caller directly instead.
    """

    def __init__(self):
        """Init for Critic2NN."""
        self._last_structure = self._last_bonded_structure = None

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

    def get_bonded_structure(self, structure: Structure, decorate: bool=False) -> StructureGraph:
        """
        Args:
            structure (Structure): Input structure
            decorate (bool, optional): Whether to decorate the structure. Defaults to False.

        Returns:
            StructureGraph: Bonded structure
        """
        from pymatgen.command_line.critic2_caller import Critic2Caller
        if structure == self._last_structure:
            sg = self._last_bonded_structure
        else:
            c2_output = Critic2Caller(structure).output
            sg = c2_output.structure_graph()
            self._last_structure = structure
            self._last_bonded_structure = sg
        if decorate:
            order_parameters = [self.get_local_order_parameters(structure, n) for n in range(len(structure))]
            sg.structure.add_site_property('order_parameters', order_parameters)
        return sg

    def get_nn_info(self, structure: Structure, n: int) -> list[dict]:
        """
        Get all near-neighbor sites as well as the associated image locations
        and weights of the site with index n in structure.

        Args:
            structure (Structure): input structure.
            n (int): index of site for which to determine near-neighbor sites.

        Returns:
            siw (list of tuples (Site, array, float)): tuples, each one
                of which represents a coordinated site, its image location, and its weight.
        """
        sg = self.get_bonded_structure(structure)
        return [{'site': connected_site.site, 'image': connected_site.jimage, 'weight': connected_site.weight, 'site_index': connected_site.index} for connected_site in sg.get_connected_sites(n)]