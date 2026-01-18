import pickle
from copy import deepcopy
import pytest
import networkx as nx
from networkx.classes import reportviews as rv
from networkx.classes.reportviews import NodeDataView
class TestNodeDataViewDefaultSetOps(TestNodeDataViewSetOps):

    @classmethod
    def setup_class(cls):
        cls.G = nx.path_graph(9)
        cls.G.nodes[3]['foo'] = 'bar'
        cls.nv = cls.G.nodes.data('foo', default=1)

    def n_its(self, nodes):
        return {(node, 'bar' if node == 3 else 1) for node in nodes}