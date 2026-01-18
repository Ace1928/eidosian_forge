from sys import version_info as _swig_python_version_info
import weakref
def SearchTrace(self, prefix):
    """
        Creates a search monitor that will trace precisely the behavior of the
        search. Use this only for low level debugging.
        """
    return _pywrapcp.Solver_SearchTrace(self, prefix)