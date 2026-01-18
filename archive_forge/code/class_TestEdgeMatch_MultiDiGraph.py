import math
from operator import eq
import networkx as nx
import networkx.algorithms.isomorphism as iso
class TestEdgeMatch_MultiDiGraph(TestEdgeMatch_MultiGraph):

    def setup_method(self):
        TestEdgeMatch_MultiGraph.setup_method(self)
        self.g1 = nx.MultiDiGraph()
        self.g2 = nx.MultiDiGraph()
        self.GM = iso.MultiDiGraphMatcher
        self.build()