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
def _igraph_from_nxgraph(graph) -> Graph:
    """Helper function that converts a networkx graph object into an igraph graph object."""
    nodes = graph.nodes(data=True)
    new_igraph = igraph.Graph()
    for node in nodes:
        new_igraph.add_vertex(name=str(node[0]), species=node[1]['specie'], coords=node[1]['coords'])
    new_igraph.add_edges([(str(edge[0]), str(edge[1])) for edge in graph.edges()])
    return new_igraph