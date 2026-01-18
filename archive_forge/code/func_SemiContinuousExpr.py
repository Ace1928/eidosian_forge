from sys import version_info as _swig_python_version_info
import weakref
def SemiContinuousExpr(self, expr, fixed_charge, step):
    """
        Semi continuous Expression (x <= 0 -> f(x) = 0; x > 0 -> f(x) = ax + b)
        a >= 0 and b >= 0
        """
    return _pywrapcp.Solver_SemiContinuousExpr(self, expr, fixed_charge, step)