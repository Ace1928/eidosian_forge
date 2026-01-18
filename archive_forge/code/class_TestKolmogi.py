import itertools
import sys
import pytest
import numpy as np
from numpy.testing import assert_
from scipy.special._testutils import FuncData
from scipy.special import kolmogorov, kolmogi, smirnov, smirnovi
from scipy.special._ufuncs import (_kolmogc, _kolmogci, _kolmogp,
class TestKolmogi:

    def test_nan(self):
        assert_(np.isnan(kolmogi(np.nan)))

    def test_basic(self):
        dataset = [(1.0, 0), (0.9639452436648751, 0.5), (0.9, 0.571173265106), (0.5, 0.8275735551899077), (0.26999967167735456, 1), (0.0006709252557796953, 2)]
        dataset = np.asarray(dataset)
        FuncData(kolmogi, dataset, (0,), 1, rtol=_rtol).check()

    def test_smallpcdf(self):
        epsilon = 0.5 ** np.arange(1, 55, 3)
        x = np.array([0.8275735551899077, 0.5345255069097583, 0.4320114038786941, 0.3736868442620478, 0.3345161714909591, 0.3057833329315859, 0.2835052890528936, 0.2655578150208676, 0.2506869966107999, 0.2380971058736669, 0.2272549289962079, 0.217787636160004, 0.2094254686862041, 0.2019676748836232, 0.1952612948137504, 0.1891874239646641, 0.1836520225050326, 0.1785795904846466])
        dataset = np.column_stack([1 - epsilon, x])
        FuncData(kolmogi, dataset, (0,), 1, rtol=_rtol).check()
        dataset = np.column_stack([epsilon, x])
        FuncData(_kolmogci, dataset, (0,), 1, rtol=_rtol).check()

    def test_smallpsf(self):
        epsilon = 0.5 ** np.arange(1, 55, 3)
        x = np.array([0.8275735551899077, 1.3163786275161036, 1.6651092133663343, 1.9525136345289607, 2.2027324540033235, 2.427292943746085, 2.6327688477341593, 2.823330050922026, 3.0018183401530627, 3.170273508408889, 3.330218444630791, 3.482825815311332, 3.629021415015205, 3.769551326282596, 3.9050272690877326, 4.035958218708255, 4.162773055788489, 4.285837174326453])
        dataset = np.column_stack([epsilon, x])
        FuncData(kolmogi, dataset, (0,), 1, rtol=_rtol).check()
        dataset = np.column_stack([1 - epsilon, x])
        FuncData(_kolmogci, dataset, (0,), 1, rtol=_rtol).check()

    def test_round_trip(self):

        def _k_ki(_p):
            return kolmogorov(kolmogi(_p))
        p = np.linspace(0.1, 1.0, 10, endpoint=True)
        dataset = np.column_stack([p, p])
        FuncData(_k_ki, dataset, (0,), 1, rtol=_rtol).check()