import contextlib
import itertools
import re
import unittest
import warnings
import numpy as np
from numba import jit, vectorize, njit
from numba.np.numpy_support import numpy_version
from numba.core import types, config
from numba.core.errors import TypingError
from numba.tests.support import TestCase, tag, skip_parfors_unsupported
from numba.np import npdatetime_helpers, numpy_support
def _get_testcases(self):
    test_cases = [np.array([DT(0, 'ns'), DT(1, 'ns'), DT(2, 'ns'), DT(3, 'ns')]), np.array([DT('2011-01-01', 'ns'), DT('1971-02-02', 'ns'), DT('1900-01-01', 'ns'), DT('2021-03-03', 'ns'), DT('2004-12-07', 'ns')]), np.array([DT('2011-01-01', 'D'), DT('1971-02-02', 'D'), DT('1900-01-01', 'D'), DT('2021-03-03', 'D'), DT('2004-12-07', 'D')]), np.array([DT('2011-01-01', 'ns'), DT('1971-02-02', 'ns'), DT('1900-01-01', 'ns'), DT('2021-03-03', 'ns'), DT('2004-12-07', 'ns'), DT('NaT', 'ns')]), np.array([DT('NaT', 'ns'), DT('2011-01-01', 'ns'), DT('1971-02-02', 'ns'), DT('1900-01-01', 'ns'), DT('2021-03-03', 'ns'), DT('2004-12-07', 'ns')]), np.array([DT('1971-02-02', 'ns'), DT('NaT', 'ns')]), np.array([DT('NaT', 'ns'), DT('NaT', 'ns'), DT('NaT', 'ns')]), np.array([TD(1, 'ns'), TD(2, 'ns'), TD(3, 'ns'), TD(4, 'ns')]), np.array([TD(1, 'D'), TD(2, 'D'), TD(3, 'D'), TD(4, 'D')]), np.array([TD('NaT', 'ns'), TD(1, 'ns'), TD(2, 'ns'), TD(3, 'ns'), TD(4, 'ns')]), np.array([TD(1, 'ns'), TD(2, 'ns'), TD(3, 'ns'), TD(4, 'ns'), TD('NaT', 'ns')]), np.array([TD('NaT', 'ns')]), np.array([TD('NaT', 'ns'), TD('NaT', 'ns'), TD('NaT', 'ns')])]
    return test_cases