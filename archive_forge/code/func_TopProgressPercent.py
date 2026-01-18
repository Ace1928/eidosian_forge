from sys import version_info as _swig_python_version_info
import weakref
def TopProgressPercent(self):
    """
        Returns a percentage representing the propress of the search before
        reaching the limits of the top-level search (can be called from a nested
        solve).
        """
    return _pywrapcp.Solver_TopProgressPercent(self)