import pytest
import networkx as nx
from .base_test import BaseTestAttributeMixing, BaseTestDegreeMixing
class TestDegreeMixingDict(BaseTestDegreeMixing):

    def test_degree_mixing_dict_undirected(self):
        d = nx.degree_mixing_dict(self.P4)
        d_result = {1: {2: 2}, 2: {1: 2, 2: 2}}
        assert d == d_result

    def test_degree_mixing_dict_undirected_normalized(self):
        d = nx.degree_mixing_dict(self.P4, normalized=True)
        d_result = {1: {2: 1.0 / 3}, 2: {1: 1.0 / 3, 2: 1.0 / 3}}
        assert d == d_result

    def test_degree_mixing_dict_directed(self):
        d = nx.degree_mixing_dict(self.D)
        print(d)
        d_result = {1: {3: 2}, 2: {1: 1, 3: 1}, 3: {}}
        assert d == d_result

    def test_degree_mixing_dict_multigraph(self):
        d = nx.degree_mixing_dict(self.M)
        d_result = {1: {2: 1}, 2: {1: 1, 3: 3}, 3: {2: 3}}
        assert d == d_result

    def test_degree_mixing_dict_weighted(self):
        d = nx.degree_mixing_dict(self.W, weight='weight')
        d_result = {0.5: {1.5: 1}, 1.5: {1.5: 6, 0.5: 1}}
        assert d == d_result