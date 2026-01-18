import logging
import unittest
from pyomo.core.base import ConcreteModel, Var, _VarData
from pyomo.common.log import LoggingIntercept
from pyomo.common.errors import ApplicationError
from pyomo.core.base.param import Param, _ParamData
from pyomo.contrib.pyros.config import (
from pyomo.contrib.pyros.util import ObjectiveType
from pyomo.opt import SolverFactory, SolverResults
from pyomo.contrib.pyros.uncertainty_sets import BoxSet
from pyomo.common.dependencies import numpy_available
class TestPositiveIntOrMinusOne(unittest.TestCase):
    """
    Test validator for -1 or positive int works as expected.
    """

    def test_positive_int_or_minus_one(self):
        """
        Test positive int or -1 validator works as expected.
        """
        standardizer_func = PositiveIntOrMinusOne()
        self.assertIs(standardizer_func(1.0), 1, msg=f'{PositiveIntOrMinusOne.__name__} does not standardize as expected.')
        self.assertEqual(standardizer_func(-1.0), -1, msg=f'{PositiveIntOrMinusOne.__name__} does not standardize as expected.')
        exc_str = 'Expected positive int or -1, but received value.*'
        with self.assertRaisesRegex(ValueError, exc_str):
            standardizer_func(1.5)
        with self.assertRaisesRegex(ValueError, exc_str):
            standardizer_func(0)