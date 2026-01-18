from sys import version_info as _swig_python_version_info
import weakref
def HasCumulVarSoftLowerBound(self, index):
    """
        Returns true if a soft lower bound has been set for a given variable
        index.
        """
    return _pywrapcp.RoutingDimension_HasCumulVarSoftLowerBound(self, index)