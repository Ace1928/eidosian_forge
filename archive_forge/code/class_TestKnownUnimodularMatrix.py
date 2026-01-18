import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_almost_equal
from ase.geometry import minkowski_reduce
from ase.cell import Cell
class TestKnownUnimodularMatrix:

    def setup_method(self):
        cell = np.array([[1, 1, 2], [0, 1, 4], [0, 0, 1]])
        unimodular = np.array([[1, 2, 2], [0, 1, 2], [0, 0, 1]])
        assert_almost_equal(np.linalg.det(unimodular), 1)
        self.lcell = unimodular.T @ cell

    @pytest.mark.parametrize('pbc', [1, True, (1, 1, 1)])
    def test_pbc(self, pbc):
        lcell = self.lcell
        rcell, op = minkowski_reduce(lcell, pbc=pbc)
        assert_almost_equal(np.linalg.det(rcell), 1)
        rdet = np.linalg.det(rcell)
        ldet = np.linalg.det(lcell)
        assert np.sign(ldet) == np.sign(rdet)

    def test_0d(self):
        lcell = self.lcell
        rcell, op = minkowski_reduce(lcell, pbc=[0, 0, 0])
        assert (rcell == lcell).all()

    @pytest.mark.parametrize('axis', range(3))
    def test_1d(self, axis):
        lcell = self.lcell
        rcell, op = minkowski_reduce(lcell, pbc=np.roll([1, 0, 0], axis))
        assert (rcell == lcell).all()
        zcell = np.zeros((3, 3))
        zcell[0] = lcell[0]
        rcell, _ = Cell(zcell).minkowski_reduce()
        assert_allclose(rcell, zcell, atol=TOL)

    @pytest.mark.parametrize('axis', range(3))
    def test_2d(self, axis):
        lcell = self.lcell
        pbc = np.roll([0, 1, 1], axis)
        rcell, op = minkowski_reduce(lcell.astype(float), pbc=pbc)
        assert (rcell[axis] == lcell[axis]).all()
        zcell = np.copy(lcell)
        zcell[axis] = 0
        rzcell, _ = Cell(zcell).minkowski_reduce()
        rcell[axis] = 0
        assert_allclose(rzcell, rcell, atol=TOL)

    def test_3d(self):
        lcell = self.lcell
        rcell, op = minkowski_reduce(lcell)
        assert_almost_equal(np.linalg.det(rcell), 1)