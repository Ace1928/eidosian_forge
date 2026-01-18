import networkx as nx
from .breadth_first_search import generic_bfs_edges
Returns a list of the best neighbors of a node.

        `v` is a node in the graph `G`.

        The "best" neighbors are chosen according to the `value`
        function (higher is better). Only the `width` best neighbors of
        `v` are returned.

        The list returned by this function is in decreasing value as
        measured by the `value` function.

        