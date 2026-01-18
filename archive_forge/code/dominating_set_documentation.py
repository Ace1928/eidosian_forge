import networkx as nx
from ...utils import not_implemented_for
from ..matching import maximal_matching
Returns the cost-effectiveness of greedily choosing the given
        node.

        `node_and_neighborhood` is a two-tuple comprising a node and its
        closed neighborhood.

        