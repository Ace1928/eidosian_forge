import os
from collections import Counter
from itertools import combinations, product
import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_equal, assert_array_equal
from scipy.spatial import distance
from scipy.stats import shapiro
from scipy.stats._sobol import _test_find_index
from scipy.stats import qmc
from scipy.stats._qmc import (
class TestMultivariateNormalQMC:

    def test_validations(self):
        message = 'Dimension of `engine` must be consistent'
        with pytest.raises(ValueError, match=message):
            qmc.MultivariateNormalQMC([0], engine=qmc.Sobol(d=2))
        message = 'Dimension of `engine` must be consistent'
        with pytest.raises(ValueError, match=message):
            qmc.MultivariateNormalQMC([0, 0, 0], engine=qmc.Sobol(d=4))
        message = '`engine` must be an instance of...'
        with pytest.raises(ValueError, match=message):
            qmc.MultivariateNormalQMC([0, 0], engine=np.random.default_rng())
        message = 'Covariance matrix not PSD.'
        with pytest.raises(ValueError, match=message):
            qmc.MultivariateNormalQMC([0, 0], [[1, 2], [2, 1]])
        message = 'Covariance matrix is not symmetric.'
        with pytest.raises(ValueError, match=message):
            qmc.MultivariateNormalQMC([0, 0], [[1, 0], [2, 1]])
        message = 'Dimension mismatch between mean and covariance.'
        with pytest.raises(ValueError, match=message):
            qmc.MultivariateNormalQMC([0], [[1, 0], [0, 1]])

    def test_MultivariateNormalQMCNonPD(self):
        engine = qmc.MultivariateNormalQMC([0, 0, 0], [[1, 0, 1], [0, 1, 1], [1, 1, 2]])
        assert engine._corr_matrix is not None

    def test_MultivariateNormalQMC(self):
        engine = qmc.MultivariateNormalQMC(mean=0, cov=5)
        samples = engine.random()
        assert_equal(samples.shape, (1, 1))
        samples = engine.random(n=5)
        assert_equal(samples.shape, (5, 1))
        engine = qmc.MultivariateNormalQMC(mean=[0, 1], cov=[[1, 0], [0, 1]])
        samples = engine.random()
        assert_equal(samples.shape, (1, 2))
        samples = engine.random(n=5)
        assert_equal(samples.shape, (5, 2))
        mean = np.array([0, 1, 2])
        cov = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        engine = qmc.MultivariateNormalQMC(mean, cov)
        samples = engine.random()
        assert_equal(samples.shape, (1, 3))
        samples = engine.random(n=5)
        assert_equal(samples.shape, (5, 3))

    def test_MultivariateNormalQMCInvTransform(self):
        engine = qmc.MultivariateNormalQMC(mean=0, cov=5, inv_transform=True)
        samples = engine.random()
        assert_equal(samples.shape, (1, 1))
        samples = engine.random(n=5)
        assert_equal(samples.shape, (5, 1))
        engine = qmc.MultivariateNormalQMC(mean=[0, 1], cov=[[1, 0], [0, 1]], inv_transform=True)
        samples = engine.random()
        assert_equal(samples.shape, (1, 2))
        samples = engine.random(n=5)
        assert_equal(samples.shape, (5, 2))
        mean = np.array([0, 1, 2])
        cov = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        engine = qmc.MultivariateNormalQMC(mean, cov, inv_transform=True)
        samples = engine.random()
        assert_equal(samples.shape, (1, 3))
        samples = engine.random(n=5)
        assert_equal(samples.shape, (5, 3))

    def test_MultivariateNormalQMCSeeded(self):
        rng = np.random.default_rng(180182791534511062935571481899241825000)
        a = rng.standard_normal((2, 2))
        A = a @ a.transpose() + np.diag(rng.random(2))
        engine = qmc.MultivariateNormalQMC(np.array([0, 0]), A, inv_transform=False, seed=rng)
        samples = engine.random(n=2)
        samples_expected = np.array([[-0.64419, -0.882413], [0.837199, 2.045301]])
        assert_allclose(samples, samples_expected, atol=0.0001)
        rng = np.random.default_rng(180182791534511062935571481899241825000)
        a = rng.standard_normal((3, 3))
        A = a @ a.transpose() + np.diag(rng.random(3))
        engine = qmc.MultivariateNormalQMC(np.array([0, 0, 0]), A, inv_transform=False, seed=rng)
        samples = engine.random(n=2)
        samples_expected = np.array([[-0.693853, -1.265338, -0.088024], [1.620193, 2.679222, 0.457343]])
        assert_allclose(samples, samples_expected, atol=0.0001)

    def test_MultivariateNormalQMCSeededInvTransform(self):
        rng = np.random.default_rng(224125808928297329711992996940871155974)
        a = rng.standard_normal((2, 2))
        A = a @ a.transpose() + np.diag(rng.random(2))
        engine = qmc.MultivariateNormalQMC(np.array([0, 0]), A, seed=rng, inv_transform=True)
        samples = engine.random(n=2)
        samples_expected = np.array([[0.682171, -3.114233], [-0.098463, 0.668069]])
        assert_allclose(samples, samples_expected, atol=0.0001)
        rng = np.random.default_rng(224125808928297329711992996940871155974)
        a = rng.standard_normal((3, 3))
        A = a @ a.transpose() + np.diag(rng.random(3))
        engine = qmc.MultivariateNormalQMC(np.array([0, 0, 0]), A, seed=rng, inv_transform=True)
        samples = engine.random(n=2)
        samples_expected = np.array([[0.988061, -1.644089, -0.877035], [-1.771731, 1.096988, 2.024744]])
        assert_allclose(samples, samples_expected, atol=0.0001)

    def test_MultivariateNormalQMCShapiro(self):
        seed = np.random.default_rng(188960007281846377164494575845971640)
        engine = qmc.MultivariateNormalQMC(mean=[0, 0], cov=[[1, 0], [0, 1]], seed=seed)
        samples = engine.random(n=256)
        assert all(np.abs(samples.mean(axis=0)) < 0.01)
        assert all(np.abs(samples.std(axis=0) - 1) < 0.01)
        for i in (0, 1):
            _, pval = shapiro(samples[:, i])
            assert pval > 0.9
        cov = np.cov(samples.transpose())
        assert np.abs(cov[0, 1]) < 0.01
        engine = qmc.MultivariateNormalQMC(mean=[1.0, 2.0], cov=[[1.5, 0.5], [0.5, 1.5]], seed=seed)
        samples = engine.random(n=256)
        assert all(np.abs(samples.mean(axis=0) - [1, 2]) < 0.01)
        assert all(np.abs(samples.std(axis=0) - np.sqrt(1.5)) < 0.01)
        for i in (0, 1):
            _, pval = shapiro(samples[:, i])
            assert pval > 0.9
        cov = np.cov(samples.transpose())
        assert np.abs(cov[0, 1] - 0.5) < 0.01

    def test_MultivariateNormalQMCShapiroInvTransform(self):
        seed = np.random.default_rng(200089821034563288698994840831440331329)
        engine = qmc.MultivariateNormalQMC(mean=[0, 0], cov=[[1, 0], [0, 1]], seed=seed, inv_transform=True)
        samples = engine.random(n=256)
        assert all(np.abs(samples.mean(axis=0)) < 0.01)
        assert all(np.abs(samples.std(axis=0) - 1) < 0.01)
        for i in (0, 1):
            _, pval = shapiro(samples[:, i])
            assert pval > 0.9
        cov = np.cov(samples.transpose())
        assert np.abs(cov[0, 1]) < 0.01
        engine = qmc.MultivariateNormalQMC(mean=[1.0, 2.0], cov=[[1.5, 0.5], [0.5, 1.5]], seed=seed, inv_transform=True)
        samples = engine.random(n=256)
        assert all(np.abs(samples.mean(axis=0) - [1, 2]) < 0.01)
        assert all(np.abs(samples.std(axis=0) - np.sqrt(1.5)) < 0.01)
        for i in (0, 1):
            _, pval = shapiro(samples[:, i])
            assert pval > 0.9
        cov = np.cov(samples.transpose())
        assert np.abs(cov[0, 1] - 0.5) < 0.01

    def test_MultivariateNormalQMCDegenerate(self):
        seed = np.random.default_rng(16320637417581448357869821654290448620)
        engine = qmc.MultivariateNormalQMC(mean=[0.0, 0.0, 0.0], cov=[[1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [1.0, 1.0, 2.0]], seed=seed)
        samples = engine.random(n=512)
        assert all(np.abs(samples.mean(axis=0)) < 0.01)
        assert np.abs(np.std(samples[:, 0]) - 1) < 0.01
        assert np.abs(np.std(samples[:, 1]) - 1) < 0.01
        assert np.abs(np.std(samples[:, 2]) - np.sqrt(2)) < 0.01
        for i in (0, 1, 2):
            _, pval = shapiro(samples[:, i])
            assert pval > 0.8
        cov = np.cov(samples.transpose())
        assert np.abs(cov[0, 1]) < 0.01
        assert np.abs(cov[0, 2] - 1) < 0.01
        assert all(np.abs(samples[:, 0] + samples[:, 1] - samples[:, 2]) < 1e-05)