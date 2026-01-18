from sys import version_info as _swig_python_version_info
import weakref
def LubyRestart(self, scale_factor):
    """
        This search monitor will restart the search periodically.
        At the iteration n, it will restart after scale_factor * Luby(n) failures
        where Luby is the Luby Strategy (i.e. 1 1 2 1 1 2 4 1 1 2 1 1 2 4 8...).
        """
    return _pywrapcp.Solver_LubyRestart(self, scale_factor)