from sys import version_info as _swig_python_version_info
import weakref
def HasCumulVarSoftUpperBound(self, index):
    """
        Returns true if a soft upper bound has been set for a given variable
        index.
        """
    return _pywrapcp.RoutingDimension_HasCumulVarSoftUpperBound(self, index)