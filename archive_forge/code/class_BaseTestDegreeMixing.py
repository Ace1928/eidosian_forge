import networkx as nx
class BaseTestDegreeMixing:

    @classmethod
    def setup_class(cls):
        cls.P4 = nx.path_graph(4)
        cls.D = nx.DiGraph()
        cls.D.add_edges_from([(0, 2), (0, 3), (1, 3), (2, 3)])
        cls.D2 = nx.DiGraph()
        cls.D2.add_edges_from([(0, 3), (1, 0), (1, 2), (2, 4), (4, 1), (4, 3), (4, 2)])
        cls.M = nx.MultiGraph()
        nx.add_path(cls.M, range(4))
        cls.M.add_edge(0, 1)
        cls.S = nx.Graph()
        cls.S.add_edges_from([(0, 0), (1, 1)])
        cls.W = nx.Graph()
        cls.W.add_edges_from([(0, 3), (1, 3), (2, 3)], weight=0.5)
        cls.W.add_edge(0, 2, weight=1)
        S1 = nx.star_graph(4)
        S2 = nx.star_graph(4)
        cls.DS = nx.disjoint_union(S1, S2)
        cls.DS.add_edge(4, 5)