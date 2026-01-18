import networkx as nx
from .isomorphvf2 import DiGraphMatcher, GraphMatcher
class TimeRespectingGraphMatcher(GraphMatcher):

    def __init__(self, G1, G2, temporal_attribute_name, delta):
        """Initialize TimeRespectingGraphMatcher.

        G1 and G2 should be nx.Graph or nx.MultiGraph instances.

        Examples
        --------
        To create a TimeRespectingGraphMatcher which checks for
        syntactic and semantic feasibility:

        >>> from networkx.algorithms import isomorphism
        >>> from datetime import timedelta
        >>> G1 = nx.Graph(nx.path_graph(4, create_using=nx.Graph()))

        >>> G2 = nx.Graph(nx.path_graph(4, create_using=nx.Graph()))

        >>> GM = isomorphism.TimeRespectingGraphMatcher(
        ...     G1, G2, "date", timedelta(days=1)
        ... )
        """
        self.temporal_attribute_name = temporal_attribute_name
        self.delta = delta
        super().__init__(G1, G2)

    def one_hop(self, Gx, Gx_node, neighbors):
        """
        Edges one hop out from a node in the mapping should be
        time-respecting with respect to each other.
        """
        dates = []
        for n in neighbors:
            if isinstance(Gx, nx.Graph):
                dates.append(Gx[Gx_node][n][self.temporal_attribute_name])
            else:
                for edge in Gx[Gx_node][n].values():
                    dates.append(edge[self.temporal_attribute_name])
        if any((x is None for x in dates)):
            raise ValueError('Datetime not supplied for at least one edge.')
        return not dates or max(dates) - min(dates) <= self.delta

    def two_hop(self, Gx, core_x, Gx_node, neighbors):
        """
        Paths of length 2 from Gx_node should be time-respecting.
        """
        return all((self.one_hop(Gx, v, [n for n in Gx[v] if n in core_x] + [Gx_node]) for v in neighbors))

    def semantic_feasibility(self, G1_node, G2_node):
        """Returns True if adding (G1_node, G2_node) is semantically
        feasible.

        Any subclass which redefines semantic_feasibility() must
        maintain the self.tests if needed, to keep the match() method
        functional. Implementations should consider multigraphs.
        """
        neighbors = [n for n in self.G1[G1_node] if n in self.core_1]
        if not self.one_hop(self.G1, G1_node, neighbors):
            return False
        if not self.two_hop(self.G1, self.core_1, G1_node, neighbors):
            return False
        return True