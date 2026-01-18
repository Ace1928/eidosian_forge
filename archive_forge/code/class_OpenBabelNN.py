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
class OpenBabelNN(NearNeighbors):
    """
    Determine near-neighbor sites and bond orders using OpenBabel API.

    NOTE: This strategy is only appropriate for molecules, and not for
    structures.
    """

    @requires(openbabel, 'BabelMolAdaptor requires openbabel to be installed with Python bindings. Please get it at http://openbabel.org (version >=3.0.0).')
    def __init__(self, order=True) -> None:
        """
        Args:
            order (bool): True if bond order should be returned as a weight, False
            if bond length should be used as a weight.
        """
        self.order = order

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
            structure: Molecule object.
            n: index of site for which to determine near neighbors.

        Returns:
            dict: representing a neighboring site and the type of
            bond present between site n and the neighboring site.
        """
        from pymatgen.io.babel import BabelMolAdaptor
        ob_mol = BabelMolAdaptor(structure).openbabel_mol
        siw = []
        site_atom = next((atom for atom in openbabel.OBMolAtomDFSIter(ob_mol) if [atom.GetX(), atom.GetY(), atom.GetZ()] == list(structure[n].coords)))
        for neighbor in openbabel.OBAtomAtomIter(site_atom):
            coords = [neighbor.GetX(), neighbor.GetY(), neighbor.GetZ()]
            site = next((a for a in structure if list(a.coords) == coords))
            index = structure.index(site)
            bond = site_atom.GetBond(neighbor)
            if self.order:
                ob_mol.PerceiveBondOrders()
                weight = bond.GetBondOrder()
            else:
                weight = bond.GetLength()
            siw.append({'site': site, 'image': (0, 0, 0), 'weight': weight, 'site_index': index})
        return siw

    def get_bonded_structure(self, structure: Structure, decorate: bool=False) -> StructureGraph:
        """
        Obtain a MoleculeGraph object using this NearNeighbor
        class. Requires the optional dependency networkx
        (pip install networkx).

        Args:
            structure: Molecule object.
            decorate (bool): whether to annotate site properties
            with order parameters using neighbors determined by
            this NearNeighbor class

        Returns:
            MoleculeGraph: object from pymatgen.analysis.graphs
        """
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