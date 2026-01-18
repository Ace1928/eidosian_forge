import pytest
import networkx as nx
from .base_test import BaseTestAttributeMixing, BaseTestDegreeMixing
class TestAttributeMixingDict(BaseTestAttributeMixing):

    def test_attribute_mixing_dict_undirected(self):
        d = nx.attribute_mixing_dict(self.G, 'fish')
        d_result = {'one': {'one': 2, 'red': 1}, 'two': {'two': 2, 'blue': 1}, 'red': {'one': 1}, 'blue': {'two': 1}}
        assert d == d_result

    def test_attribute_mixing_dict_directed(self):
        d = nx.attribute_mixing_dict(self.D, 'fish')
        d_result = {'one': {'one': 1, 'red': 1}, 'two': {'two': 1, 'blue': 1}, 'red': {}, 'blue': {}}
        assert d == d_result

    def test_attribute_mixing_dict_multigraph(self):
        d = nx.attribute_mixing_dict(self.M, 'fish')
        d_result = {'one': {'one': 4}, 'two': {'two': 2}}
        assert d == d_result