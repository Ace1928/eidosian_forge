from sys import version_info as _swig_python_version_info
import weakref
def GetCumulVarSoftUpperBound(self, index):
    """
        Returns the soft upper bound of a cumul variable for a given variable
        index. The "hard" upper bound of the variable is returned if no soft upper
        bound has been set.
        """
    return _pywrapcp.RoutingDimension_GetCumulVarSoftUpperBound(self, index)