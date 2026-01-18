import hashlib
import pickle
import sys
import warnings
import numpy as np
import pytest
from numpy.testing import (
from numpy.random import MT19937, PCG64
from numpy import random
class TestSingleEltArrayInput:

    def setup_method(self):
        self.argOne = np.array([2])
        self.argTwo = np.array([3])
        self.argThree = np.array([4])
        self.tgtShape = (1,)

    def test_one_arg_funcs(self):
        funcs = (random.exponential, random.standard_gamma, random.chisquare, random.standard_t, random.pareto, random.weibull, random.power, random.rayleigh, random.poisson, random.zipf, random.geometric, random.logseries)
        probfuncs = (random.geometric, random.logseries)
        for func in funcs:
            if func in probfuncs:
                out = func(np.array([0.5]))
            else:
                out = func(self.argOne)
            assert_equal(out.shape, self.tgtShape)

    def test_two_arg_funcs(self):
        funcs = (random.uniform, random.normal, random.beta, random.gamma, random.f, random.noncentral_chisquare, random.vonmises, random.laplace, random.gumbel, random.logistic, random.lognormal, random.wald, random.binomial, random.negative_binomial)
        probfuncs = (random.binomial, random.negative_binomial)
        for func in funcs:
            if func in probfuncs:
                argTwo = np.array([0.5])
            else:
                argTwo = self.argTwo
            out = func(self.argOne, argTwo)
            assert_equal(out.shape, self.tgtShape)
            out = func(self.argOne[0], argTwo)
            assert_equal(out.shape, self.tgtShape)
            out = func(self.argOne, argTwo[0])
            assert_equal(out.shape, self.tgtShape)

    def test_three_arg_funcs(self):
        funcs = [random.noncentral_f, random.triangular, random.hypergeometric]
        for func in funcs:
            out = func(self.argOne, self.argTwo, self.argThree)
            assert_equal(out.shape, self.tgtShape)
            out = func(self.argOne[0], self.argTwo, self.argThree)
            assert_equal(out.shape, self.tgtShape)
            out = func(self.argOne, self.argTwo[0], self.argThree)
            assert_equal(out.shape, self.tgtShape)