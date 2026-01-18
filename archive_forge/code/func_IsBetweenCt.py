from sys import version_info as _swig_python_version_info
import weakref
def IsBetweenCt(self, expr, l, u, b):
    """ b == (l <= expr <= u)"""
    return _pywrapcp.Solver_IsBetweenCt(self, expr, l, u, b)