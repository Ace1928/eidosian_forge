import itertools
import sys
import pytest
import numpy as np
from numpy.testing import assert_
from scipy.special._testutils import FuncData
from scipy.special import kolmogorov, kolmogi, smirnov, smirnovi
from scipy.special._ufuncs import (_kolmogc, _kolmogci, _kolmogp,
def _smc_smci(n, p):
    return _smirnovc(n, _smirnovci(n, p))