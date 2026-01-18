from sys import version_info as _swig_python_version_info
import weakref
def GetCumulVarSoftLowerBoundCoefficient(self, index):
    """
        Returns the cost coefficient of the soft lower bound of a cumul variable
        for a given variable index. If no soft lower bound has been set, 0 is
        returned.
        """
    return _pywrapcp.RoutingDimension_GetCumulVarSoftLowerBoundCoefficient(self, index)