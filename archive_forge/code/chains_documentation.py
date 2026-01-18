import networkx as nx
from networkx.utils import not_implemented_for
Generate the chain starting from the given nontree edge.

        `G` is a DFS cycle graph as constructed by
        :func:`_dfs_cycle_graph`. The edge (`u`, `v`) is a nontree edge
        that begins a chain. `visited` is a set representing the nodes
        in `G` that have already been visited.

        This function yields the edges in an initial segment of the
        fundamental cycle of `G` starting with the nontree edge (`u`,
        `v`) that includes all the edges up until the first node that
        appears in `visited`. The tree edges are given by the 'parent'
        node attribute. The `visited` set is updated to add each node in
        an edge yielded by this function.

        