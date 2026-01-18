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
class CovalentBondNN(NearNeighbors):
    """
    Determine near-neighbor sites and bond orders using built-in
    pymatgen.Molecule CovalentBond functionality.

    NOTE: This strategy is only appropriate for molecules, and not for
    structures.
    """

    def __init__(self, tol: float=0.2, order=True) -> None:
        """
        Args:
            tol (float): Tolerance for covalent bond checking.
            order (bool): If True (default), this class will compute bond
                orders. If False, bond lengths will be computed.
        """
        self.tol = tol
        self.order = order
        self.bonds = None

    @property
    def structures_allowed(self) -> bool:
        """
        Boolean property: can this NearNeighbors class be used with Structure
        objects?
        """
        return False

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
        return False

    def get_nn_info(self, structure: Structure, n: int):
        """
        Get all near-neighbor sites and weights (orders) of bonds for a given
        atom.

        Args:
            structure: input Molecule.
            n: index of site for which to determine near neighbors.

        Returns:
            [dict] representing a neighboring site and the type of
            bond present between site n and the neighboring site.
        """
        self.bonds = bonds = structure.get_covalent_bonds(tol=self.tol)
        siw = []
        for bond in bonds:
            capture_bond = False
            if bond.site1 == structure[n]:
                site = bond.site2
                capture_bond = True
            elif bond.site2 == structure[n]:
                site = bond.site1
                capture_bond = True
            if capture_bond:
                index = structure.index(site)
                weight = bond.get_bond_order() if self.order else bond.length
                siw.append({'site': site, 'image': (0, 0, 0), 'weight': weight, 'site_index': index})
        return siw

    def get_bonded_structure(self, structure: Structure, decorate: bool=False) -> MoleculeGraph:
        """
        Obtain a MoleculeGraph object using this NearNeighbor class.

        Args:
            structure: Molecule object.
            decorate (bool): whether to annotate site properties
            with order parameters using neighbors determined by
            this NearNeighbor class

        Returns:
            MoleculeGraph: object from pymatgen.analysis.graphs
        """
        from pymatgen.analysis.graphs import MoleculeGraph
        if decorate:
            order_parameters = [self.get_local_order_parameters(structure, n) for n in range(len(structure))]
            structure.add_site_property('order_parameters', order_parameters)
        return MoleculeGraph.from_local_env_strategy(structure, self)

    def get_nn_shell_info(self, structure: Structure, site_idx, shell):
        """Get a certain nearest neighbor shell for a certain site.

        Determines all non-backtracking paths through the neighbor network
        computed by `get_nn_info`. The weight is determined by multiplying
        the weight of the neighbor at each hop through the network. For
        example, a 2nd-nearest-neighbor that has a weight of 1 from its
        1st-nearest-neighbor and weight 0.5 from the original site will
        be assigned a weight of 0.5.

        As this calculation may involve computing the nearest neighbors of
        atoms multiple times, the calculation starts by computing all of the
        neighbor info and then calling `_get_nn_shell_info`. If you are likely
        to call this method for more than one site, consider calling `get_all_nn`
        first and then calling this protected method yourself.

        Args:
            structure (Molecule): Input structure
            site_idx (int): index of site for which to determine neighbor
                information.
            shell (int): Which neighbor shell to retrieve (1 == 1st NN shell)

        Returns:
            list of dictionaries. Each entry in the list is information about
                a certain neighbor in the structure, in the same format as
                `get_nn_info`.
        """
        all_nn_info = self.get_all_nn_info(structure)
        sites = self._get_nn_shell_info(structure, all_nn_info, site_idx, shell)
        output = []
        for info in sites:
            orig_site = structure[info['site_index']]
            info['site'] = Site(orig_site.species, orig_site._coords, properties=orig_site.properties)
            output.append(info)
        return output