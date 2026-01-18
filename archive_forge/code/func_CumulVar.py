from sys import version_info as _swig_python_version_info
import weakref
def CumulVar(self, index):
    """
        Get the cumul, transit and slack variables for the given node (given as
        int64_t var index).
        """
    return _pywrapcp.RoutingDimension_CumulVar(self, index)