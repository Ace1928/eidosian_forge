import itertools
from typing import (
import networkx as nx
import rustworkx as rx
import numpy as np
import pennylane as qml
from pennylane.ops import Hamiltonian
Calculates the squared inner portion of the Hamiltonian in :func:`net_flow_constraint`.


    For a given :math:`i`, this function returns:

    .. math::

        \left((d_{i}^{\rm out} - d_{i}^{\rm in})\mathbb{I} -
        \sum_{j, (i, j) \in E} Z_{ij} + \sum_{j, (j, i) \in E} Z_{ji} \right)^{2}.

    Args:
        graph (nx.DiGraph or rx.PyDiGraph): the directed graph specifying possible edges
        node: a fixed node

    Returns:
        qml.Hamiltonian: The inner part of the net-flow constraint Hamiltonian.
    