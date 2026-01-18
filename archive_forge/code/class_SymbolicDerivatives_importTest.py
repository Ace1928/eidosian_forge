import pyomo.common.unittest as unittest
from pyomo.common.errors import DeveloperError, NondifferentiableError
from pyomo.environ import (
from pyomo.core.expr.calculus.diff_with_sympy import differentiate
from pyomo.core.expr.sympy_tools import (
class SymbolicDerivatives_importTest(unittest.TestCase):

    def test_sympy_avail_flag(self):
        if sympy_available:
            import sympy
        else:
            with self.assertRaises(ImportError):
                import sympy