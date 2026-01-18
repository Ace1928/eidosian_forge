from sys import version_info as _swig_python_version_info
import weakref
def GetMutableGlobalCumulLPOptimizer(self, dimension):
    """
        Returns the global/local dimension cumul optimizer for a given dimension,
        or nullptr if there is none.
        """
    return _pywrapcp.RoutingModel_GetMutableGlobalCumulLPOptimizer(self, dimension)