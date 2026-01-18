from random import Random
import pytest
import networkx as nx
from networkx import convert_node_labels_to_integers as cnlti
from networkx.algorithms.distance_measures import _extrema_bounding
class TestWeightedDistance:

    def setup_method(self):
        G = nx.Graph()
        G.add_edge(0, 1, weight=0.6, cost=0.6, high_cost=6)
        G.add_edge(0, 2, weight=0.2, cost=0.2, high_cost=2)
        G.add_edge(2, 3, weight=0.1, cost=0.1, high_cost=1)
        G.add_edge(2, 4, weight=0.7, cost=0.7, high_cost=7)
        G.add_edge(2, 5, weight=0.9, cost=0.9, high_cost=9)
        G.add_edge(1, 5, weight=0.3, cost=0.3, high_cost=3)
        self.G = G
        self.weight_fn = lambda v, u, e: 2

    def test_eccentricity_weight_None(self):
        assert nx.eccentricity(self.G, 1, weight=None) == 3
        e = nx.eccentricity(self.G, weight=None)
        assert e[1] == 3
        e = nx.eccentricity(self.G, v=1, weight=None)
        assert e == 3
        e = nx.eccentricity(self.G, v=[1, 1], weight=None)
        assert e[1] == 3
        e = nx.eccentricity(self.G, v=[1, 2], weight=None)
        assert e[1] == 3

    def test_eccentricity_weight_attr(self):
        assert nx.eccentricity(self.G, 1, weight='weight') == 1.5
        e = nx.eccentricity(self.G, weight='weight')
        assert e == nx.eccentricity(self.G, weight='cost') != nx.eccentricity(self.G, weight='high_cost')
        assert e[1] == 1.5
        e = nx.eccentricity(self.G, v=1, weight='weight')
        assert e == 1.5
        e = nx.eccentricity(self.G, v=[1, 1], weight='weight')
        assert e[1] == 1.5
        e = nx.eccentricity(self.G, v=[1, 2], weight='weight')
        assert e[1] == 1.5

    def test_eccentricity_weight_fn(self):
        assert nx.eccentricity(self.G, 1, weight=self.weight_fn) == 6
        e = nx.eccentricity(self.G, weight=self.weight_fn)
        assert e[1] == 6
        e = nx.eccentricity(self.G, v=1, weight=self.weight_fn)
        assert e == 6
        e = nx.eccentricity(self.G, v=[1, 1], weight=self.weight_fn)
        assert e[1] == 6
        e = nx.eccentricity(self.G, v=[1, 2], weight=self.weight_fn)
        assert e[1] == 6

    def test_diameter_weight_None(self):
        assert nx.diameter(self.G, weight=None) == 3

    def test_diameter_weight_attr(self):
        assert nx.diameter(self.G, weight='weight') == nx.diameter(self.G, weight='cost') == 1.6 != nx.diameter(self.G, weight='high_cost')

    def test_diameter_weight_fn(self):
        assert nx.diameter(self.G, weight=self.weight_fn) == 6

    def test_radius_weight_None(self):
        assert pytest.approx(nx.radius(self.G, weight=None)) == 2

    def test_radius_weight_attr(self):
        assert pytest.approx(nx.radius(self.G, weight='weight')) == pytest.approx(nx.radius(self.G, weight='cost')) == 0.9 != nx.radius(self.G, weight='high_cost')

    def test_radius_weight_fn(self):
        assert nx.radius(self.G, weight=self.weight_fn) == 4

    def test_periphery_weight_None(self):
        for v in set(nx.periphery(self.G, weight=None)):
            assert nx.eccentricity(self.G, v, weight=None) == nx.diameter(self.G, weight=None)

    def test_periphery_weight_attr(self):
        periphery = set(nx.periphery(self.G, weight='weight'))
        assert periphery == set(nx.periphery(self.G, weight='cost')) == set(nx.periphery(self.G, weight='high_cost'))
        for v in periphery:
            assert nx.eccentricity(self.G, v, weight='high_cost') != nx.eccentricity(self.G, v, weight='weight') == nx.eccentricity(self.G, v, weight='cost') == nx.diameter(self.G, weight='weight') == nx.diameter(self.G, weight='cost') != nx.diameter(self.G, weight='high_cost')
            assert nx.eccentricity(self.G, v, weight='high_cost') == nx.diameter(self.G, weight='high_cost')

    def test_periphery_weight_fn(self):
        for v in set(nx.periphery(self.G, weight=self.weight_fn)):
            assert nx.eccentricity(self.G, v, weight=self.weight_fn) == nx.diameter(self.G, weight=self.weight_fn)

    def test_center_weight_None(self):
        for v in set(nx.center(self.G, weight=None)):
            assert pytest.approx(nx.eccentricity(self.G, v, weight=None)) == nx.radius(self.G, weight=None)

    def test_center_weight_attr(self):
        center = set(nx.center(self.G, weight='weight'))
        assert center == set(nx.center(self.G, weight='cost')) != set(nx.center(self.G, weight='high_cost'))
        for v in center:
            assert nx.eccentricity(self.G, v, weight='high_cost') != pytest.approx(nx.eccentricity(self.G, v, weight='weight')) == pytest.approx(nx.eccentricity(self.G, v, weight='cost')) == nx.radius(self.G, weight='weight') == nx.radius(self.G, weight='cost') != nx.radius(self.G, weight='high_cost')
            assert nx.eccentricity(self.G, v, weight='high_cost') == nx.radius(self.G, weight='high_cost')

    def test_center_weight_fn(self):
        for v in set(nx.center(self.G, weight=self.weight_fn)):
            assert nx.eccentricity(self.G, v, weight=self.weight_fn) == nx.radius(self.G, weight=self.weight_fn)

    def test_bound_diameter_weight_None(self):
        assert nx.diameter(self.G, usebounds=True, weight=None) == 3

    def test_bound_diameter_weight_attr(self):
        assert nx.diameter(self.G, usebounds=True, weight='high_cost') != nx.diameter(self.G, usebounds=True, weight='weight') == nx.diameter(self.G, usebounds=True, weight='cost') == 1.6 != nx.diameter(self.G, usebounds=True, weight='high_cost')
        assert nx.diameter(self.G, usebounds=True, weight='high_cost') == nx.diameter(self.G, usebounds=True, weight='high_cost')

    def test_bound_diameter_weight_fn(self):
        assert nx.diameter(self.G, usebounds=True, weight=self.weight_fn) == 6

    def test_bound_radius_weight_None(self):
        assert pytest.approx(nx.radius(self.G, usebounds=True, weight=None)) == 2

    def test_bound_radius_weight_attr(self):
        assert nx.radius(self.G, usebounds=True, weight='high_cost') != pytest.approx(nx.radius(self.G, usebounds=True, weight='weight')) == pytest.approx(nx.radius(self.G, usebounds=True, weight='cost')) == 0.9 != nx.radius(self.G, usebounds=True, weight='high_cost')
        assert nx.radius(self.G, usebounds=True, weight='high_cost') == nx.radius(self.G, usebounds=True, weight='high_cost')

    def test_bound_radius_weight_fn(self):
        assert nx.radius(self.G, usebounds=True, weight=self.weight_fn) == 4

    def test_bound_periphery_weight_None(self):
        result = {1, 3, 4}
        assert set(nx.periphery(self.G, usebounds=True, weight=None)) == result

    def test_bound_periphery_weight_attr(self):
        result = {4, 5}
        assert set(nx.periphery(self.G, usebounds=True, weight='weight')) == set(nx.periphery(self.G, usebounds=True, weight='cost')) == result

    def test_bound_periphery_weight_fn(self):
        result = {1, 3, 4}
        assert set(nx.periphery(self.G, usebounds=True, weight=self.weight_fn)) == result

    def test_bound_center_weight_None(self):
        result = {0, 2, 5}
        assert set(nx.center(self.G, usebounds=True, weight=None)) == result

    def test_bound_center_weight_attr(self):
        result = {0}
        assert set(nx.center(self.G, usebounds=True, weight='weight')) == set(nx.center(self.G, usebounds=True, weight='cost')) == result

    def test_bound_center_weight_fn(self):
        result = {0, 2, 5}
        assert set(nx.center(self.G, usebounds=True, weight=self.weight_fn)) == result