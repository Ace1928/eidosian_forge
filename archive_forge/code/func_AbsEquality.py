from sys import version_info as _swig_python_version_info
import weakref
def AbsEquality(self, var, abs_var):
    """ Creates the constraint abs(var) == abs_var."""
    return _pywrapcp.Solver_AbsEquality(self, var, abs_var)