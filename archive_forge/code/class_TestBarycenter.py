from random import Random
import pytest
import networkx as nx
from networkx import convert_node_labels_to_integers as cnlti
from networkx.algorithms.distance_measures import _extrema_bounding
class TestBarycenter:
    """Test :func:`networkx.algorithms.distance_measures.barycenter`."""

    def barycenter_as_subgraph(self, g, **kwargs):
        """Return the subgraph induced on the barycenter of g"""
        b = nx.barycenter(g, **kwargs)
        assert isinstance(b, list)
        assert set(b) <= set(g)
        return g.subgraph(b)

    def test_must_be_connected(self):
        pytest.raises(nx.NetworkXNoPath, nx.barycenter, nx.empty_graph(5))

    def test_sp_kwarg(self):
        K_5 = nx.complete_graph(5)
        sp = dict(nx.shortest_path_length(K_5))
        assert nx.barycenter(K_5, sp=sp) == list(K_5)
        for u, v, data in K_5.edges.data():
            data['weight'] = 1
        pytest.raises(ValueError, nx.barycenter, K_5, sp=sp, weight='weight')
        del sp[0][1]
        pytest.raises(nx.NetworkXNoPath, nx.barycenter, K_5, sp=sp)

    def test_trees(self):
        """The barycenter of a tree is a single vertex or an edge.

        See [West01]_, p. 78.
        """
        prng = Random(3735928559)
        for i in range(50):
            RT = nx.random_labeled_tree(prng.randint(1, 75), seed=prng)
            b = self.barycenter_as_subgraph(RT)
            if len(b) == 2:
                assert b.size() == 1
            else:
                assert len(b) == 1
                assert b.size() == 0

    def test_this_one_specific_tree(self):
        """Test the tree pictured at the bottom of [West01]_, p. 78."""
        g = nx.Graph({'a': ['b'], 'b': ['a', 'x'], 'x': ['b', 'y'], 'y': ['x', 'z'], 'z': ['y', 0, 1, 2, 3, 4], 0: ['z'], 1: ['z'], 2: ['z'], 3: ['z'], 4: ['z']})
        b = self.barycenter_as_subgraph(g, attr='barycentricity')
        assert list(b) == ['z']
        assert not b.edges
        expected_barycentricity = {0: 23, 1: 23, 2: 23, 3: 23, 4: 23, 'a': 35, 'b': 27, 'x': 21, 'y': 17, 'z': 15}
        for node, barycentricity in expected_barycentricity.items():
            assert g.nodes[node]['barycentricity'] == barycentricity
        for edge in g.edges:
            g.edges[edge]['weight'] = 2
        b = self.barycenter_as_subgraph(g, weight='weight', attr='barycentricity2')
        assert list(b) == ['z']
        assert not b.edges
        for node, barycentricity in expected_barycentricity.items():
            assert g.nodes[node]['barycentricity2'] == barycentricity * 2