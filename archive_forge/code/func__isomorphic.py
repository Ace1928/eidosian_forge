from __future__ import annotations
import copy
import logging
import os.path
import subprocess
import warnings
from collections import defaultdict, namedtuple
from itertools import combinations
from operator import itemgetter
from shutil import which
from typing import TYPE_CHECKING, Any, Callable, cast
import networkx as nx
import networkx.algorithms.isomorphism as iso
import numpy as np
from monty.dev import deprecated
from monty.json import MSONable
from networkx.drawing.nx_agraph import write_dot
from networkx.readwrite import json_graph
from scipy.spatial import KDTree
from scipy.stats import describe
from pymatgen.core import Lattice, Molecule, PeriodicSite, Structure
from pymatgen.core.structure import FunctionalGroups
from pymatgen.util.coord import lattice_points_in_supercell
from pymatgen.vis.structure_vtk import EL_COLORS
def _isomorphic(frag1: nx.Graph, frag2: nx.Graph) -> bool:
    """
    Helper function to check if two graph objects are isomorphic, using igraph if
    if is available and networkx if it is not.
    """
    f1_nodes = frag1.nodes(data=True)
    f2_nodes = frag2.nodes(data=True)
    if len(f1_nodes) != len(f2_nodes):
        return False
    f1_edges = frag1.edges()
    f2_edges = frag2.edges()
    if len(f1_edges) != len(f2_edges):
        return False
    f1_comp_dict = {}
    f2_comp_dict = {}
    for node in f1_nodes:
        if node[1]['specie'] not in f1_comp_dict:
            f1_comp_dict[node[1]['specie']] = 1
        else:
            f1_comp_dict[node[1]['specie']] += 1
    for node in f2_nodes:
        if node[1]['specie'] not in f2_comp_dict:
            f2_comp_dict[node[1]['specie']] = 1
        else:
            f2_comp_dict[node[1]['specie']] += 1
    if f1_comp_dict != f2_comp_dict:
        return False
    if igraph is not None:
        ifrag1 = _igraph_from_nxgraph(frag1)
        ifrag2 = _igraph_from_nxgraph(frag2)
        return ifrag1.isomorphic_vf2(ifrag2, node_compat_fn=_compare)
    nm = iso.categorical_node_match('specie', 'ERROR')
    return nx.is_isomorphic(frag1.to_undirected(), frag2.to_undirected(), node_match=nm)