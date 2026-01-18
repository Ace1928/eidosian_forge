import pytest
import numpy as np
from scipy.optimize import quadratic_assignment, OptimizeWarning
from scipy.optimize._qap import _calc_score as _score
from numpy.testing import assert_equal, assert_, assert_warns
class TestQAPOnce:

    def setup_method(self):
        np.random.seed(0)

    def test_common_input_validation(self):
        with pytest.raises(ValueError, match='`A` must be square'):
            quadratic_assignment(np.random.random((3, 4)), np.random.random((3, 3)))
        with pytest.raises(ValueError, match='`B` must be square'):
            quadratic_assignment(np.random.random((3, 3)), np.random.random((3, 4)))
        with pytest.raises(ValueError, match='`A` and `B` must have exactly two'):
            quadratic_assignment(np.random.random((3, 3, 3)), np.random.random((3, 3, 3)))
        with pytest.raises(ValueError, match='`A` and `B` matrices must be of equal size'):
            quadratic_assignment(np.random.random((3, 3)), np.random.random((4, 4)))
        _rm = _range_matrix
        with pytest.raises(ValueError, match='`partial_match` can have only as many seeds as'):
            quadratic_assignment(np.identity(3), np.identity(3), options={'partial_match': _rm(5, 2)})
        with pytest.raises(ValueError, match='`partial_match` must have two columns'):
            quadratic_assignment(np.identity(3), np.identity(3), options={'partial_match': _range_matrix(2, 3)})
        with pytest.raises(ValueError, match='`partial_match` must have exactly two'):
            quadratic_assignment(np.identity(3), np.identity(3), options={'partial_match': np.random.rand(3, 2, 2)})
        with pytest.raises(ValueError, match='`partial_match` must contain only pos'):
            quadratic_assignment(np.identity(3), np.identity(3), options={'partial_match': -1 * _range_matrix(2, 2)})
        with pytest.raises(ValueError, match='`partial_match` entries must be less than number'):
            quadratic_assignment(np.identity(5), np.identity(5), options={'partial_match': 2 * _range_matrix(4, 2)})
        with pytest.raises(ValueError, match='`partial_match` column entries must be unique'):
            quadratic_assignment(np.identity(3), np.identity(3), options={'partial_match': np.ones((2, 2))})