from sys import version_info as _swig_python_version_info
import weakref
def ConditionalExpression(self, condition, expr, unperformed_value):
    """ Conditional Expr condition ? expr : unperformed_value"""
    return _pywrapcp.Solver_ConditionalExpression(self, condition, expr, unperformed_value)