from itertools import chain, combinations, product
import pytest
import networkx as nx
class TestMultiDiGraph_DAGLCA(TestDAGLCA):

    @classmethod
    def setup_class(cls):
        cls.DG = nx.MultiDiGraph()
        nx.add_path(cls.DG, (0, 1, 2, 3))
        nx.add_path(cls.DG, (0, 1, 2, 3))
        nx.add_path(cls.DG, (0, 4, 3))
        nx.add_path(cls.DG, (0, 5, 6, 8, 3))
        nx.add_path(cls.DG, (5, 7, 8))
        cls.DG.add_edge(6, 2)
        cls.DG.add_edge(7, 2)
        cls.root_distance = nx.shortest_path_length(cls.DG, source=0)
        cls.gold = {(1, 1): 1, (1, 2): 1, (1, 3): 1, (1, 4): 0, (1, 5): 0, (1, 6): 0, (1, 7): 0, (1, 8): 0, (2, 2): 2, (2, 3): 2, (2, 4): 0, (2, 5): 5, (2, 6): 6, (2, 7): 7, (2, 8): 7, (3, 3): 3, (3, 4): 4, (3, 5): 5, (3, 6): 6, (3, 7): 7, (3, 8): 8, (4, 4): 4, (4, 5): 0, (4, 6): 0, (4, 7): 0, (4, 8): 0, (5, 5): 5, (5, 6): 5, (5, 7): 5, (5, 8): 5, (6, 6): 6, (6, 7): 5, (6, 8): 6, (7, 7): 7, (7, 8): 7, (8, 8): 8}
        cls.gold.update((((0, n), 0) for n in cls.DG))