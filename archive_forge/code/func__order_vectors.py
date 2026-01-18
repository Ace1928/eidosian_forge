from __future__ import annotations
import itertools
import logging
from typing import TYPE_CHECKING
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch
from monty.json import MSONable, jsanitize
from networkx.algorithms.components import is_connected
from networkx.algorithms.traversal import bfs_tree
from pymatgen.analysis.chemenv.connectivity.environment_nodes import EnvironmentNode
from pymatgen.analysis.chemenv.utils.chemenv_errors import ChemenvError
from pymatgen.analysis.chemenv.utils.graph_utils import get_delta
from pymatgen.analysis.chemenv.utils.math_utils import get_linearly_independent_vectors
@staticmethod
def _order_vectors(vectors):
    """Orders vectors.

        First, each vector is made such that the first non-zero dimension is positive.
        Example: a periodicity vector [0, -1, 1] is transformed to [0, 1, -1].
        Then vectors are ordered based on their first element, then (if the first element
        is identical) based on their second element, then (if the first and second element
        are identical) based on their third element and so on ...
        Example: [[1, 1, 0], [0, 1, -1], [0, 1, 1]] is ordered as [[0, 1, -1], [0, 1, 1], [1, 1, 0]]
        """
    for ipv, pv in enumerate(vectors):
        non_zeros = np.nonzero(pv)[0]
        if pv[non_zeros[0]] < 0 < len(non_zeros):
            vectors[ipv] = -pv
    return sorted(vectors, key=lambda x: x.tolist())