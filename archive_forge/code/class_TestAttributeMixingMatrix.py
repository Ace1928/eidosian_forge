import pytest
import networkx as nx
from .base_test import BaseTestAttributeMixing, BaseTestDegreeMixing
class TestAttributeMixingMatrix(BaseTestAttributeMixing):

    def test_attribute_mixing_matrix_undirected(self):
        mapping = {'one': 0, 'two': 1, 'red': 2, 'blue': 3}
        a_result = np.array([[2, 0, 1, 0], [0, 2, 0, 1], [1, 0, 0, 0], [0, 1, 0, 0]])
        a = nx.attribute_mixing_matrix(self.G, 'fish', mapping=mapping, normalized=False)
        np.testing.assert_equal(a, a_result)
        a = nx.attribute_mixing_matrix(self.G, 'fish', mapping=mapping)
        np.testing.assert_equal(a, a_result / a_result.sum())

    def test_attribute_mixing_matrix_directed(self):
        mapping = {'one': 0, 'two': 1, 'red': 2, 'blue': 3}
        a_result = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0]])
        a = nx.attribute_mixing_matrix(self.D, 'fish', mapping=mapping, normalized=False)
        np.testing.assert_equal(a, a_result)
        a = nx.attribute_mixing_matrix(self.D, 'fish', mapping=mapping)
        np.testing.assert_equal(a, a_result / a_result.sum())

    def test_attribute_mixing_matrix_multigraph(self):
        mapping = {'one': 0, 'two': 1, 'red': 2, 'blue': 3}
        a_result = np.array([[4, 0, 0, 0], [0, 2, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        a = nx.attribute_mixing_matrix(self.M, 'fish', mapping=mapping, normalized=False)
        np.testing.assert_equal(a, a_result)
        a = nx.attribute_mixing_matrix(self.M, 'fish', mapping=mapping)
        np.testing.assert_equal(a, a_result / a_result.sum())

    def test_attribute_mixing_matrix_negative(self):
        mapping = {-2: 0, -3: 1, -4: 2}
        a_result = np.array([[4.0, 1.0, 1.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        a = nx.attribute_mixing_matrix(self.N, 'margin', mapping=mapping, normalized=False)
        np.testing.assert_equal(a, a_result)
        a = nx.attribute_mixing_matrix(self.N, 'margin', mapping=mapping)
        np.testing.assert_equal(a, a_result / float(a_result.sum()))

    def test_attribute_mixing_matrix_float(self):
        mapping = {0.5: 1, 1.5: 0}
        a_result = np.array([[6.0, 1.0], [1.0, 0.0]])
        a = nx.attribute_mixing_matrix(self.F, 'margin', mapping=mapping, normalized=False)
        np.testing.assert_equal(a, a_result)
        a = nx.attribute_mixing_matrix(self.F, 'margin', mapping=mapping)
        np.testing.assert_equal(a, a_result / a_result.sum())