import networkx as nx
from .isomorphvf2 import DiGraphMatcher, GraphMatcher
class TimeRespectingDiGraphMatcher(DiGraphMatcher):

    def __init__(self, G1, G2, temporal_attribute_name, delta):
        """Initialize TimeRespectingDiGraphMatcher.

        G1 and G2 should be nx.DiGraph or nx.MultiDiGraph instances.

        Examples
        --------
        To create a TimeRespectingDiGraphMatcher which checks for
        syntactic and semantic feasibility:

        >>> from networkx.algorithms import isomorphism
        >>> from datetime import timedelta
        >>> G1 = nx.DiGraph(nx.path_graph(4, create_using=nx.DiGraph()))

        >>> G2 = nx.DiGraph(nx.path_graph(4, create_using=nx.DiGraph()))

        >>> GM = isomorphism.TimeRespectingDiGraphMatcher(
        ...     G1, G2, "date", timedelta(days=1)
        ... )
        """
        self.temporal_attribute_name = temporal_attribute_name
        self.delta = delta
        super().__init__(G1, G2)

    def get_pred_dates(self, Gx, Gx_node, core_x, pred):
        """
        Get the dates of edges from predecessors.
        """
        pred_dates = []
        if isinstance(Gx, nx.DiGraph):
            for n in pred:
                pred_dates.append(Gx[n][Gx_node][self.temporal_attribute_name])
        else:
            for n in pred:
                for edge in Gx[n][Gx_node].values():
                    pred_dates.append(edge[self.temporal_attribute_name])
        return pred_dates

    def get_succ_dates(self, Gx, Gx_node, core_x, succ):
        """
        Get the dates of edges to successors.
        """
        succ_dates = []
        if isinstance(Gx, nx.DiGraph):
            for n in succ:
                succ_dates.append(Gx[Gx_node][n][self.temporal_attribute_name])
        else:
            for n in succ:
                for edge in Gx[Gx_node][n].values():
                    succ_dates.append(edge[self.temporal_attribute_name])
        return succ_dates

    def one_hop(self, Gx, Gx_node, core_x, pred, succ):
        """
        The ego node.
        """
        pred_dates = self.get_pred_dates(Gx, Gx_node, core_x, pred)
        succ_dates = self.get_succ_dates(Gx, Gx_node, core_x, succ)
        return self.test_one(pred_dates, succ_dates) and self.test_two(pred_dates, succ_dates)

    def two_hop_pred(self, Gx, Gx_node, core_x, pred):
        """
        The predecessors of the ego node.
        """
        return all((self.one_hop(Gx, p, core_x, self.preds(Gx, core_x, p), self.succs(Gx, core_x, p, Gx_node)) for p in pred))

    def two_hop_succ(self, Gx, Gx_node, core_x, succ):
        """
        The successors of the ego node.
        """
        return all((self.one_hop(Gx, s, core_x, self.preds(Gx, core_x, s, Gx_node), self.succs(Gx, core_x, s)) for s in succ))

    def preds(self, Gx, core_x, v, Gx_node=None):
        pred = [n for n in Gx.predecessors(v) if n in core_x]
        if Gx_node:
            pred.append(Gx_node)
        return pred

    def succs(self, Gx, core_x, v, Gx_node=None):
        succ = [n for n in Gx.successors(v) if n in core_x]
        if Gx_node:
            succ.append(Gx_node)
        return succ

    def test_one(self, pred_dates, succ_dates):
        """
        Edges one hop out from Gx_node in the mapping should be
        time-respecting with respect to each other, regardless of
        direction.
        """
        time_respecting = True
        dates = pred_dates + succ_dates
        if any((x is None for x in dates)):
            raise ValueError('Date or datetime not supplied for at least one edge.')
        dates.sort()
        if 0 < len(dates) and (not dates[-1] - dates[0] <= self.delta):
            time_respecting = False
        return time_respecting

    def test_two(self, pred_dates, succ_dates):
        """
        Edges from a dual Gx_node in the mapping should be ordered in
        a time-respecting manner.
        """
        time_respecting = True
        pred_dates.sort()
        succ_dates.sort()
        if 0 < len(succ_dates) and 0 < len(pred_dates) and (succ_dates[0] < pred_dates[-1]):
            time_respecting = False
        return time_respecting

    def semantic_feasibility(self, G1_node, G2_node):
        """Returns True if adding (G1_node, G2_node) is semantically
        feasible.

        Any subclass which redefines semantic_feasibility() must
        maintain the self.tests if needed, to keep the match() method
        functional. Implementations should consider multigraphs.
        """
        pred, succ = ([n for n in self.G1.predecessors(G1_node) if n in self.core_1], [n for n in self.G1.successors(G1_node) if n in self.core_1])
        if not self.one_hop(self.G1, G1_node, self.core_1, pred, succ):
            return False
        if not self.two_hop_pred(self.G1, G1_node, self.core_1, pred):
            return False
        if not self.two_hop_succ(self.G1, G1_node, self.core_1, succ):
            return False
        return True