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
class TestNormalQMC:

    def test_NormalQMC(self):
        engine = qmc.MultivariateNormalQMC(mean=np.zeros(1))
        samples = engine.random()
        assert_equal(samples.shape, (1, 1))
        samples = engine.random(n=5)
        assert_equal(samples.shape, (5, 1))
        engine = qmc.MultivariateNormalQMC(mean=np.zeros(2))
        samples = engine.random()
        assert_equal(samples.shape, (1, 2))
        samples = engine.random(n=5)
        assert_equal(samples.shape, (5, 2))

    def test_NormalQMCInvTransform(self):
        engine = qmc.MultivariateNormalQMC(mean=np.zeros(1), inv_transform=True)
        samples = engine.random()
        assert_equal(samples.shape, (1, 1))
        samples = engine.random(n=5)
        assert_equal(samples.shape, (5, 1))
        engine = qmc.MultivariateNormalQMC(mean=np.zeros(2), inv_transform=True)
        samples = engine.random()
        assert_equal(samples.shape, (1, 2))
        samples = engine.random(n=5)
        assert_equal(samples.shape, (5, 2))

    def test_NormalQMCSeeded(self):
        seed = np.random.default_rng(274600237797326520096085022671371676017)
        engine = qmc.MultivariateNormalQMC(mean=np.zeros(2), inv_transform=False, seed=seed)
        samples = engine.random(n=2)
        samples_expected = np.array([[-0.932001, -0.522923], [-1.477655, 0.846851]])
        assert_allclose(samples, samples_expected, atol=0.0001)
        seed = np.random.default_rng(274600237797326520096085022671371676017)
        engine = qmc.MultivariateNormalQMC(mean=np.zeros(3), inv_transform=False, seed=seed)
        samples = engine.random(n=2)
        samples_expected = np.array([[-0.932001, -0.522923, 0.036578], [-1.778011, 0.912428, -0.065421]])
        assert_allclose(samples, samples_expected, atol=0.0001)
        seed = np.random.default_rng(274600237797326520096085022671371676017)
        base_engine = qmc.Sobol(4, scramble=True, seed=seed)
        engine = qmc.MultivariateNormalQMC(mean=np.zeros(3), inv_transform=False, engine=base_engine, seed=seed)
        samples = engine.random(n=2)
        samples_expected = np.array([[-0.932001, -0.522923, 0.036578], [-1.778011, 0.912428, -0.065421]])
        assert_allclose(samples, samples_expected, atol=0.0001)

    def test_NormalQMCSeededInvTransform(self):
        seed = np.random.default_rng(288527772707286126646493545351112463929)
        engine = qmc.MultivariateNormalQMC(mean=np.zeros(2), seed=seed, inv_transform=True)
        samples = engine.random(n=2)
        samples_expected = np.array([[-0.913237, -0.964026], [0.255904, 0.003068]])
        assert_allclose(samples, samples_expected, atol=0.0001)
        seed = np.random.default_rng(288527772707286126646493545351112463929)
        engine = qmc.MultivariateNormalQMC(mean=np.zeros(3), seed=seed, inv_transform=True)
        samples = engine.random(n=2)
        samples_expected = np.array([[-0.913237, -0.964026, 0.355501], [0.699261, 2.90213, -0.6418]])
        assert_allclose(samples, samples_expected, atol=0.0001)

    def test_other_engine(self):
        for d in (0, 1, 2):
            base_engine = qmc.Sobol(d=d, scramble=False)
            engine = qmc.MultivariateNormalQMC(mean=np.zeros(d), engine=base_engine, inv_transform=True)
            samples = engine.random()
            assert_equal(samples.shape, (1, d))

    def test_NormalQMCShapiro(self):
        rng = np.random.default_rng(13242)
        engine = qmc.MultivariateNormalQMC(mean=np.zeros(2), seed=rng)
        samples = engine.random(n=256)
        assert all(np.abs(samples.mean(axis=0)) < 0.01)
        assert all(np.abs(samples.std(axis=0) - 1) < 0.01)
        for i in (0, 1):
            _, pval = shapiro(samples[:, i])
            assert pval > 0.9
        cov = np.cov(samples.transpose())
        assert np.abs(cov[0, 1]) < 0.01

    def test_NormalQMCShapiroInvTransform(self):
        rng = np.random.default_rng(32344554)
        engine = qmc.MultivariateNormalQMC(mean=np.zeros(2), inv_transform=True, seed=rng)
        samples = engine.random(n=256)
        assert all(np.abs(samples.mean(axis=0)) < 0.01)
        assert all(np.abs(samples.std(axis=0) - 1) < 0.01)
        for i in (0, 1):
            _, pval = shapiro(samples[:, i])
            assert pval > 0.9
        cov = np.cov(samples.transpose())
        assert np.abs(cov[0, 1]) < 0.01